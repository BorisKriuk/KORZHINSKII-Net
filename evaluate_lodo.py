"""evaluate_lodo.py — leave-one-deposit-out (or K-fold) evaluation.

Compares PINN vs several baseline classifiers on PR-AUC, with each known
mine held out (LODO) or with K-fold CV (faster).

Baselines:
  - proxy-only : M = k_mod_x(x) * src_z(z), NO training. Sanity check
                 that quantifies how much of PINN's score comes from the
                 priors alone vs from the PDE-constrained network.
  - LogisticRegression
  - RandomForest
  - ExtraTrees
  - GradientBoosting
  - KNN
  - SVM (RBF)
  - MLP

Negatives sampling modes:
  --neg-mode random  : uniform x in [0.5, AR-0.5], at least r_inner from any
                       mine; uniform z in [0.05, 0.95]. (Old behavior.)
  --neg-mode hard    : x sampled in a RING [r_inner, r_outer] around the
                       nearest mine; z sampled from the proxy peak distribution
                       (matched to positives). Removes geometric leak so the
                       baseline classifiers cannot trivially ace the task.

CV modes:
  --kfold 0          : classic LODO (1 retraining per positive). Default.
  --kfold N (N>=2)   : N-fold CV (only N retrainings). Much faster.

Fair comparison:
  By default baseline classifiers DO NOT see the proxy features
  (k_mod_x, src_z) -- those are the same priors that PINN uses, so giving
  them to the baselines causes circular evaluation. Use
  --baseline-use-proxy to enable them and reproduce the leaky setup.

  In addition, positives' z is sampled from the SAME distribution used for
  negatives (proxy peak with small jitter), instead of being pinned to the
  argmax of src_z. This removes a vertical-coordinate leak that allowed
  tree-based models to memorise "z == constant -> mine".

Usage:
    # all sites, fair 5-fold + hard negatives
    python evaluate_lodo.py --sites all --epochs 800 \
        --neg-mode hard --r-inner 0.4 --r-outer 2.0 --kfold 5 --tag fair_5fold_all
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from pinn import (
    PINN, compute_M, lith_at, build_column, project_mines,
    build_proxy_modulators, load_heat_flow_target,
    train_one, AR, N_LITH, OTH, DEVICE,
    DEPOSIT_TYPE_BY_SITE, DEFAULT_DEPOSIT_TYPE,
    auto_z_prior_from_src,
)
from sites import SITES

# ----- defaults ---------------------------------------------------------------
N_NEG          = 200
LODO_EPOCHS    = 800
LODO_LR        = 2e-3
LODO_NCOL      = 1500
EXCL_RADIUS    = 0.4
HARD_R_INNER   = 0.4
HARD_R_OUTER   = 2.0
W_ZPRIOR       = 0.5


# ============================================================ data helpers
def get_site_data(site_key):
    s = SITES[site_key]
    data_dir = Path("data") / site_key
    column = build_column(data_dir, s["fallback_column"])
    mines = project_mines(data_dir, s["lat"], s["lon"])
    T_bottom, hf = load_heat_flow_target(
        data_dir, site_key, s.get("heat_flow_mwm2", 55.0))
    deposit_type = s.get("deposit_type") \
                   or DEPOSIT_TYPE_BY_SITE.get(site_key, DEFAULT_DEPOSIT_TYPE)
    k_mod_x, src_z, proxy_meta = build_proxy_modulators(
        data_dir, s["lat"], s["lon"], deposit_type, column)
    srcz_grid = torch.linspace(0, 1, src_z.numel(), device=DEVICE)
    z_prior = auto_z_prior_from_src(srcz_grid, src_z)
    return {"site": s, "column": column, "mines": mines,
            "T_bottom": T_bottom, "hf": hf,
            "deposit_type": deposit_type,
            "k_mod_x": k_mod_x, "src_z": src_z,
            "proxy_meta": proxy_meta, "z_prior": z_prior}


# ----- random (old) negatives -------------------------------------------------
def sample_negatives_random(mines, n_neg=N_NEG,
                            exclude_radius=EXCL_RADIUS, seed=42):
    rng = np.random.default_rng(seed)
    pts, tries = [], 0
    while len(pts) < n_neg and tries < n_neg * 50:
        x = float(rng.uniform(0.5, AR - 0.5))
        z = float(rng.uniform(0.05, 0.95))
        tries += 1
        if any(abs(x - mx) < exclude_radius for mx in mines):
            continue
        pts.append((x, z))
    if len(pts) < n_neg:
        raise RuntimeError(
            f"random-negative sampling: got {len(pts)}/{n_neg}")
    return pts


# ----- hard (ring) negatives --------------------------------------------------
def sample_negatives_hard(mines, src_z, n_neg=N_NEG,
                          r_inner=HARD_R_INNER, r_outer=HARD_R_OUTER,
                          z_match_proxy=True, seed=42):
    rng = np.random.default_rng(seed)
    mines_arr = np.asarray(mines, dtype=np.float32)

    src_z_np = src_z.detach().cpu().numpy() \
               if hasattr(src_z, "detach") else np.asarray(src_z)
    src_z_np = np.clip(src_z_np, 1e-6, None)
    p_z = src_z_np / src_z_np.sum()
    z_grid = np.linspace(0.0, 1.0, len(p_z))

    pts, tries, max_tries = [], 0, n_neg * 500
    while len(pts) < n_neg and tries < max_tries:
        tries += 1
        x = float(rng.uniform(0.0, AR))
        d_min = float(np.min(np.abs(mines_arr - x)))
        if d_min < r_inner or d_min > r_outer:
            continue
        if z_match_proxy:
            z = float(rng.choice(z_grid, p=p_z))
            z = float(np.clip(z + rng.normal(0.0, 0.04), 0.05, 0.95))
        else:
            z = float(rng.uniform(0.05, 0.95))
        pts.append((x, z))

    if len(pts) < n_neg:
        raise RuntimeError(
            f"hard-negative sampling produced only {len(pts)}/{n_neg}. "
            f"Mines x-range = [{mines_arr.min():.2f}, {mines_arr.max():.2f}], "
            f"r_inner={r_inner}, r_outer={r_outer}. "
            f"Try increasing r_outer or decreasing r_inner."
        )
    return pts


def sample_negatives(mode, mines, src_z, n_neg, r_inner, r_outer,
                     exclude_radius, seed=42):
    if mode == "random":
        return sample_negatives_random(mines, n_neg=n_neg,
                                       exclude_radius=exclude_radius,
                                       seed=seed)
    elif mode == "hard":
        return sample_negatives_hard(mines, src_z, n_neg=n_neg,
                                     r_inner=r_inner, r_outer=r_outer,
                                     z_match_proxy=True, seed=seed)
    else:
        raise ValueError(f"unknown neg-mode: {mode}")


# ============================================================ utilities
def best_z_for_x(model, x_val, column, n_z=80):
    z_grid = torch.linspace(0.05, 0.95, n_z, device=DEVICE)
    x_t = torch.full_like(z_grid, float(x_val))
    M = compute_M(model, x_t, z_grid, column).cpu().numpy()
    iz = int(np.argmax(M))
    return float(z_grid[iz].item()), float(M[iz])


def find_all_contacts_z(column):
    out = []
    n = len(column)
    for i in range(n - 1):
        if column[i][2] == OTH:
            continue
        j = i + 1
        while j < n and column[j][2] == OTH:
            j += 1
        if j >= n:
            continue
        out.append(0.5 * (column[i][1] + column[j][0]))
    return out


def sample_pos_z_like_neg(n, src_z_np, seed):
    """Sample positives' z from the same proxy-peak distribution used for
    negatives, with the same Gaussian jitter and clipping."""
    rng = np.random.default_rng(seed)
    src = np.clip(src_z_np, 1e-6, None)
    p_z = src / src.sum()
    z_grid = np.linspace(0.0, 1.0, len(p_z))
    z = rng.choice(z_grid, size=n, p=p_z)
    z = np.clip(z + rng.normal(0.0, 0.04, size=n), 0.05, 0.95)
    return z.astype(np.float32)


# ============================================================ feature builder
def features(xs, zs, column, k_mod_x, src_z, contacts, use_proxy=False):
    """Lithology one-hot + z + distance to nearest contact.
    Optionally (use_proxy=True): k_mod_x(x), src_z(z) appended.
    """
    xs = np.asarray(xs, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)
    n = len(xs)
    L = lith_at(torch.as_tensor(xs), torch.as_tensor(zs), column).cpu().numpy()
    oh = np.zeros((n, N_LITH), dtype=np.float32)
    oh[np.arange(n), L] = 1.0
    if contacts:
        cs = np.asarray(contacts, dtype=np.float32)
        dz = np.min(np.abs(zs[:, None] - cs[None, :]), axis=1)
    else:
        dz = np.full(n, 1.0, dtype=np.float32)
    cols = [oh, zs.reshape(-1, 1), dz.reshape(-1, 1)]
    if use_proxy:
        kmod_grid = np.linspace(0, AR, k_mod_x.numel())
        srcz_grid = np.linspace(0, 1, src_z.numel())
        kmod = np.interp(xs, kmod_grid, k_mod_x.cpu().numpy())
        srcz = np.interp(zs, srcz_grid, src_z.cpu().numpy())
        cols += [kmod.reshape(-1, 1), srcz.reshape(-1, 1)]
    return np.hstack(cols).astype(np.float32)


# ============================================================ fold builder
def make_folds(n_pos, k_folds, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_pos)
    rng.shuffle(idx)
    if k_folds is None or k_folds <= 0 or k_folds >= n_pos:
        return [[i] for i in range(n_pos)], "LODO"
    return [list(map(int, f)) for f in np.array_split(idx, k_folds)], \
           f"{k_folds}-fold"


# ============================================================ PINN CV
def run_pinn_cv(site_key, data, neg_xz, epochs=LODO_EPOCHS,
                k_folds=0, seed=42):
    mines = data["mines"]
    column = data["column"]
    z_prior = data["z_prior"]
    print(f"  z_prior (auto from proxy) = {z_prior}")

    n_pos = len(mines)
    n_neg = len(neg_xz)
    pos_scores = np.zeros(n_pos, dtype=np.float32)
    neg_scores_acc = np.zeros(n_neg, dtype=np.float32)
    neg_scores_count = 0

    neg_x = np.array([p[0] for p in neg_xz], dtype=np.float32)
    neg_z = np.array([p[1] for p in neg_xz], dtype=np.float32)

    folds, mode_str = make_folds(n_pos, k_folds, seed)
    print(f"  PINN CV mode: {mode_str}  ({len(folds)} trainings)")

    for fi, held_idx in enumerate(folds):
        held_mines = [mines[j] for j in held_idx]
        train_mines = [m for j, m in enumerate(mines) if j not in held_idx]
        print(f"  fold {fi+1}/{len(folds)}: hold out "
              f"{len(held_idx)} mine(s)  "
              f"x={[f'{m:.2f}' for m in held_mines]}")

        model, _, _ = train_one(
            site_key, seed=0, epochs=epochs, n_col=LODO_NCOL, lr=LODO_LR,
            mines_train=train_mines, mines_val=[],
            column=column, z_prior=z_prior, w_zprior=W_ZPRIOR,
            T_bottom=data["T_bottom"],
            k_mod_x=data["k_mod_x"], src_z=data["src_z"],
            verbose=False)

        for j in held_idx:
            _, M_held = best_z_for_x(model, mines[j], column)
            pos_scores[j] = M_held

        x_t = torch.tensor(neg_x, device=DEVICE)
        z_t = torch.tensor(neg_z, device=DEVICE)
        neg_scores_acc += compute_M(
            model, x_t, z_t, column).cpu().numpy()
        neg_scores_count += 1

        m_held_mean = float(np.mean([pos_scores[j] for j in held_idx]))
        m_neg_mean = float(neg_scores_acc.mean() / neg_scores_count)
        print(f"    -> M_held(mean)={m_held_mean:.3f}  "
              f"M_neg(mean)={m_neg_mean:.3f}")

    neg_scores = neg_scores_acc / max(neg_scores_count, 1)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    s = np.concatenate([pos_scores, neg_scores])
    return y, s, pos_scores, neg_scores


# ============================================================ proxy-only baseline
def run_proxy_only(data, neg_xz, pos_zs):
    """No training: M_proxy(x,z) = k_mod_x(x) * src_z(z).
    Sanity check that quantifies how much of PINN's signal is already in
    the priors before any PDE solving.
    """
    mines = data["mines"]
    k_mod_x = data["k_mod_x"].cpu().numpy()
    src_z = data["src_z"].cpu().numpy()
    kmod_grid = np.linspace(0, AR, len(k_mod_x))
    srcz_grid = np.linspace(0, 1, len(src_z))

    pos_xs = np.array(mines, dtype=np.float32)
    neg_xs = np.array([p[0] for p in neg_xz], dtype=np.float32)
    neg_zs = np.array([p[1] for p in neg_xz], dtype=np.float32)

    pos_kx = np.interp(pos_xs, kmod_grid, k_mod_x)
    pos_sz = np.interp(pos_zs, srcz_grid, src_z)
    neg_kx = np.interp(neg_xs, kmod_grid, k_mod_x)
    neg_sz = np.interp(neg_zs, srcz_grid, src_z)

    pos_scores = (pos_kx * pos_sz).astype(np.float32)
    neg_scores = (neg_kx * neg_sz).astype(np.float32)
    return pos_scores, neg_scores


# ============================================================ baseline CV
def run_baseline_cv(data, neg_xz, classifier_factory,
                    k_folds=0, seed=42, use_proxy=False,
                    pos_zs=None):
    mines = data["mines"]
    column = data["column"]
    contacts = find_all_contacts_z(column)

    pos_xs = np.array(mines, dtype=np.float32)
    neg_xs = np.array([p[0] for p in neg_xz], dtype=np.float32)
    neg_zs = np.array([p[1] for p in neg_xz], dtype=np.float32)

    F_pos = features(pos_xs, pos_zs, column,
                     data["k_mod_x"], data["src_z"], contacts,
                     use_proxy=use_proxy)
    F_neg = features(neg_xs, neg_zs, column,
                     data["k_mod_x"], data["src_z"], contacts,
                     use_proxy=use_proxy)

    n_pos = len(mines)
    n_neg = len(F_neg)
    pos_scores = np.zeros(n_pos)
    neg_scores_acc = np.zeros(n_neg)
    neg_scores_count = 0

    folds, _ = make_folds(n_pos, k_folds, seed)

    for fi, held_idx in enumerate(folds):
        idx_tr = [j for j in range(n_pos) if j not in held_idx]
        X_tr = np.vstack([F_pos[idx_tr], F_neg])
        y_tr = np.concatenate([np.ones(len(idx_tr)),
                               np.zeros(len(F_neg))])
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = classifier_factory()
        clf.fit(X_tr_s, y_tr)

        Xh = scaler.transform(F_pos[held_idx])
        ph = clf.predict_proba(Xh)[:, 1]
        for k, j in enumerate(held_idx):
            pos_scores[j] = ph[k]

        Xn = scaler.transform(F_neg)
        neg_scores_acc += clf.predict_proba(Xn)[:, 1]
        neg_scores_count += 1

    neg_scores = neg_scores_acc / max(neg_scores_count, 1)
    return pos_scores, neg_scores


def median_rank(pos_scores, neg_scores):
    ranks = []
    for p in pos_scores:
        ranks.append(float((neg_scores >= p).sum() + 1) /
                     float(len(neg_scores) + 1))
    return float(np.mean(ranks))


# ============================================================ baseline registry
def get_baseline_factories():
    """Returns dict {name: factory()}.
    All classifiers expose .predict_proba (SVC has probability=True).
    """
    return {
        "LR": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "RF": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            random_state=0, n_jobs=-1),
        "ET": lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            random_state=0, n_jobs=-1),
        "GB": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=0),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "SVM": lambda: SVC(C=1.0, gamma="scale", probability=True,
                           random_state=0),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 64), max_iter=2000,
            early_stopping=True, random_state=0),
    }


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["all"],
                    help="site keys, or 'all' to use every site in SITES")
    ap.add_argument("--epochs", type=int, default=LODO_EPOCHS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--neg-mode", choices=["random", "hard"], default="hard")
    ap.add_argument("--r-inner", type=float, default=HARD_R_INNER)
    ap.add_argument("--r-outer", type=float, default=HARD_R_OUTER)
    ap.add_argument("--excl-radius", type=float, default=EXCL_RADIUS)
    ap.add_argument("--kfold", type=int, default=0,
                    help="0 = classic LODO; N>=2 = N-fold CV.")
    ap.add_argument("--baseline-use-proxy", action="store_true")
    ap.add_argument("--baseline-pos-z-pinned", action="store_true")
    ap.add_argument("--skip-pinn", action="store_true",
                    help="Run only baselines (useful for quick re-eval).")
    ap.add_argument("--baselines", nargs="+", default=None,
                    help="subset of baselines to run; default = all. "
                         "Options: LR RF ET GB KNN SVM MLP")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    # Expand "all"
    if len(args.sites) == 1 and args.sites[0].lower() == "all":
        args.sites = list(SITES.keys())
    print(f">>> sites = {args.sites}")

    # Select baselines
    all_factories = get_baseline_factories()
    if args.baselines is None:
        baseline_names = list(all_factories.keys())
    else:
        unknown = [b for b in args.baselines if b not in all_factories]
        if unknown:
            raise SystemExit(f"unknown baselines: {unknown}. "
                             f"Available: {list(all_factories.keys())}")
        baseline_names = args.baselines
    print(f">>> baselines = {baseline_names} (+ proxy-only)")

    print(f">>> negatives mode = {args.neg_mode}")
    if args.neg_mode == "hard":
        print(f"    ring: r_inner={args.r_inner}  r_outer={args.r_outer}")
    else:
        print(f"    excl_radius={args.excl_radius}")
    if args.kfold and args.kfold >= 2:
        print(f">>> CV mode: {args.kfold}-fold")
    else:
        print(">>> CV mode: classic LODO")
    print(f">>> baseline proxy features: "
          f"{'ON (LEAKY)' if args.baseline_use_proxy else 'OFF (fair)'}")
    print(f">>> baseline positive z: "
          f"{'pinned to argmax(src_z) (LEAKY)' if args.baseline_pos_z_pinned else 'sampled like negatives (fair)'}")

    rows = []
    detailed = {}
    for site_key in args.sites:
        if site_key not in SITES:
            print(f"\n!! unknown site '{site_key}', skipping")
            continue
        print(f"\n========== EVAL: {site_key} ==========")
        try:
            data = get_site_data(site_key)
        except Exception as e:
            print(f"  !! could not load site data: {e}")
            continue
        mines = data["mines"]
        if len(mines) < 4:
            print(f"  only {len(mines)} mines — too few; skipping")
            continue

        try:
            neg_xz = sample_negatives(
                mode=args.neg_mode,
                mines=mines, src_z=data["src_z"],
                n_neg=args.n_neg,
                r_inner=args.r_inner, r_outer=args.r_outer,
                exclude_radius=args.excl_radius,
                seed=args.seed,
            )
        except RuntimeError as e:
            print(f"  !! negative sampling failed: {e}")
            continue

        prev = len(mines) / (len(mines) + len(neg_xz))
        print(f"  {len(mines)} positives, {len(neg_xz)} negatives "
              f"(prevalence={prev:.3f})")
        nx = np.array([p[0] for p in neg_xz])
        nz = np.array([p[1] for p in neg_xz])
        ma = np.asarray(mines)
        d_min = np.min(np.abs(nx[:, None] - ma[None, :]), axis=1)
        print(f"  negatives x in [{nx.min():.2f}, {nx.max():.2f}], "
              f"z in [{nz.min():.2f}, {nz.max():.2f}], "
              f"dist-to-nearest-mine in [{d_min.min():.2f}, "
              f"{d_min.max():.2f}] (mean {d_min.mean():.2f})")

        # positives' z (shared between proxy-only and baselines)
        src_z_np = data["src_z"].cpu().numpy()
        if args.baseline_pos_z_pinned:
            srcz_grid = np.linspace(0, 1, len(src_z_np))
            z_pos = float(srcz_grid[np.argmax(src_z_np)])
            pos_zs = np.full(len(mines), z_pos, dtype=np.float32)
        else:
            pos_zs = sample_pos_z_like_neg(len(mines), src_z_np,
                                           seed=args.seed + 1)

        site_results = {}

        # --- PINN ---
        if not args.skip_pinn:
            print("\n--- PINN CV ---")
            y_p, s_p, pp, np_ = run_pinn_cv(
                site_key, data, neg_xz,
                epochs=args.epochs, k_folds=args.kfold, seed=args.seed)
            ap_p = average_precision_score(y_p, s_p)
            rk_p = median_rank(pp, np_)
            print(f"  PINN  PR-AUC = {ap_p:.3f}   mean_rank = {rk_p:.3f}")
            site_results["PINN"] = (ap_p, rk_p, pp, np_)

        # --- proxy-only sanity baseline ---
        print("\n--- proxy-only (no training) ---")
        pp_x, np_x = run_proxy_only(data, neg_xz, pos_zs)
        y_x = np.concatenate([np.ones(len(pp_x)), np.zeros(len(np_x))])
        s_x = np.concatenate([pp_x, np_x])
        ap_x = average_precision_score(y_x, s_x)
        rk_x = median_rank(pp_x, np_x)
        print(f"  PROXY PR-AUC = {ap_x:.3f}   mean_rank = {rk_x:.3f}")
        site_results["PROXY"] = (ap_x, rk_x, pp_x, np_x)

        # --- learned baselines ---
        for bname in baseline_names:
            print(f"\n--- {bname} CV ---")
            try:
                pb, nb = run_baseline_cv(
                    data, neg_xz,
                    all_factories[bname],
                    k_folds=args.kfold, seed=args.seed,
                    use_proxy=args.baseline_use_proxy,
                    pos_zs=pos_zs)
            except Exception as e:
                print(f"  !! {bname} failed: {e}")
                continue
            y_b = np.concatenate([np.ones(len(pb)), np.zeros(len(nb))])
            s_b = np.concatenate([pb, nb])
            ap_b = average_precision_score(y_b, s_b)
            rk_b = median_rank(pb, nb)
            print(f"  {bname:<5s} PR-AUC = {ap_b:.3f}   "
                  f"mean_rank = {rk_b:.3f}")
            site_results[bname] = (ap_b, rk_b, pb, nb)

        # row for table
        row = {
            "site": site_key, "n_mines": len(mines), "n_neg": len(neg_xz),
            "prevalence": prev,
            "neg_mode": args.neg_mode,
            "r_inner": args.r_inner if args.neg_mode == "hard" else None,
            "r_outer": args.r_outer if args.neg_mode == "hard" else None,
            "kfold": args.kfold,
            "baseline_use_proxy": bool(args.baseline_use_proxy),
            "baseline_pos_z_pinned": bool(args.baseline_pos_z_pinned),
        }
        for k, (ap_v, rk_v, _, _) in site_results.items():
            row[f"{k}_PR_AUC"] = ap_v
            row[f"{k}_mean_rank"] = rk_v
        rows.append(row)
        detailed[site_key] = {
            k: {"pos": v[2].tolist() if hasattr(v[2], "tolist") else list(v[2]),
                "neg": v[3].tolist() if hasattr(v[3], "tolist") else list(v[3])}
            for k, v in site_results.items()
        }

    # ----- summary table -----
    cv_label = f"{args.kfold}-fold" if args.kfold and args.kfold >= 2 \
               else "LODO"
    if not rows:
        print("\nno sites evaluated.")
        return

    # column order
    model_order = []
    if not args.skip_pinn:
        model_order.append("PINN")
    model_order.append("PROXY")
    model_order.extend(baseline_names)
    model_order = [m for m in model_order
                   if any(f"{m}_PR_AUC" in r for r in rows)]

    print("\n" + "=" * (28 + 9 * len(model_order)))
    print(f"NEG MODE: {args.neg_mode}    CV: {cv_label}    "
          f"baseline_proxy={'ON' if args.baseline_use_proxy else 'OFF'}    "
          f"pos_z_pinned={'ON' if args.baseline_pos_z_pinned else 'OFF'}")

    # PR-AUC table
    header = f"{'site':<14} {'n_pos':>5} {'prev':>5}  " + \
             "".join(f"{m+'_AP':>9}" for m in model_order)
    print("\n[PR-AUC]")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['site']:<14} {r['n_mines']:>5d} {r['prevalence']:>5.2f}  "
        for m in model_order:
            v = r.get(f"{m}_PR_AUC", float("nan"))
            line += f"{v:>9.3f}"
        print(line)

    # mean rank table
    print("\n[mean fractional rank (lower = better, 0 = perfect)]")
    header2 = f"{'site':<14} {'n_pos':>5} {'prev':>5}  " + \
              "".join(f"{m+'_rk':>9}" for m in model_order)
    print(header2)
    print("-" * len(header2))
    for r in rows:
        line = f"{r['site']:<14} {r['n_mines']:>5d} {r['prevalence']:>5.2f}  "
        for m in model_order:
            v = r.get(f"{m}_mean_rank", float("nan"))
            line += f"{v:>9.3f}"
        print(line)

    # mean across sites
    print("\n[MEAN across sites]")
    print(f"{'metric':<10}" + "".join(f"{m:>9}" for m in model_order))
    means_ap = {m: np.nanmean([r.get(f"{m}_PR_AUC", np.nan)
                               for r in rows]) for m in model_order}
    means_rk = {m: np.nanmean([r.get(f"{m}_mean_rank", np.nan)
                               for r in rows]) for m in model_order}
    line = f"{'PR-AUC':<10}" + "".join(f"{means_ap[m]:>9.3f}"
                                        for m in model_order)
    print(line)
    line = f"{'rank':<10}" + "".join(f"{means_rk[m]:>9.3f}"
                                      for m in model_order)
    print(line)
    print("=" * len(header))

    Path("output").mkdir(exist_ok=True)
    if args.tag:
        tag = args.tag
    else:
        proxy_tag = "leaky" if args.baseline_use_proxy else "fair"
        tag = f"{args.neg_mode}_{cv_label}_{proxy_tag}"
    out = Path("output") / f"lodo_summary_{tag}.json"
    out.write_text(json.dumps({
        "config": {
            "sites": args.sites,
            "neg_mode": args.neg_mode,
            "r_inner": args.r_inner,
            "r_outer": args.r_outer,
            "excl_radius": args.excl_radius,
            "n_neg": args.n_neg,
            "epochs": args.epochs,
            "kfold": args.kfold,
            "cv_mode": cv_label,
            "seed": args.seed,
            "baseline_use_proxy": bool(args.baseline_use_proxy),
            "baseline_pos_z_pinned": bool(args.baseline_pos_z_pinned),
            "models": model_order,
        },
        "rows": rows,
        "mean_PR_AUC": means_ap,
        "mean_rank": means_rk,
        "detailed": detailed},
        indent=2, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()