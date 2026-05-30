"""Mineral Prospectivity PINN — multi-site (v6).

Changes vs v5:
  - auto_z_prior_from_src(): if site has no z_prior set, derive one from
    the argmax of the precomputed src_z proxy bump. Fixes Udokan/Natalka/
    Mirny where the column ran but the peak depth wasn't pinned.
"""
import argparse, json, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sites import SITES, DEFAULT, HF_REF_MWM2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- domain geometry / dimensionless numbers ----------------------------
AR      = 10.0
LX_M    = 50_000.0
RA      = 50.0
PE_T    = 30.0
PE_C    = 100.0
N_LITH  = 5
DIP_AMP = 0.08
LITH_NAMES = ["sediment", "basalt", "intrusive", "evaporite", "other"]
SED, BAS, INT, EVA, OTH = 0, 1, 2, 3, 4

# -------- modulator grids (fixed for state_dict stability) -------------------
X_GRID_N = 80
Z_GRID_N = 80
BW_X     = 0.4
BW_Z     = 0.04
KMOD_FLOOR    = 0.5
KMOD_GAIN     = 1.5
KMOD_GAIN_EQ  = 0.75
SRC_FLOOR     = 1.0
SRC_GAIN      = 3.0
SRC_GAIN_DEEP = 2.0

# threshold above which src_z is considered "non-flat" enough to derive a prior
AUTO_ZPRIOR_THRESH = 1.5

DEPOSIT_TYPE_BY_SITE = {
    "norilsk":      "magmatic_sulfide",
    "pechenga":     "magmatic_sulfide",
    "talnakh":      "magmatic_sulfide",
    "udokan":       "sediment_cu",
    "dzhezkazgan":  "sediment_cu",
    "sukhoi_log":   "orogenic_au",
    "natalka":      "orogenic_au",
    "muruntau":     "orogenic_au",
    "mirny":        "kimberlite",
    "aikhal":       "kimberlite",
}
DEFAULT_DEPOSIT_TYPE = "hydrothermal"


# ---------------------------------------------------------------- lith helpers
def classify_lith(s):
    s = (s or "").lower()
    if any(k in s for k in ["basalt", "volcanic", "tuff", "lava"]):
        return BAS
    if any(k in s for k in ["intrusive", "gabbro", "dolerite", "diorite",
                            "granite", "norite", "pluton", "kimberlite"]):
        return INT
    if any(k in s for k in ["evaporite", "anhydrite", "gypsum",
                            "halite", "salt"]):
        return EVA
    if any(k in s for k in ["sandstone", "shale", "limestone", "mudstone",
                            "siltstone", "carbonate", "dolomite", "argillite",
                            "marl", "conglomerate"]):
        return SED
    return OTH


def build_column(data_dir, fallback):
    layers = []
    try:
        data = json.loads((data_dir / "units.json").read_text())
        units = data.get("success", {}).get("data", []) or []
        for u in units:
            t = u.get("max_thick") or u.get("min_thick") or 200.0
            try:
                t = float(t)
            except Exception:
                t = 200.0
            if t <= 0:
                t = 200.0
            lith = u.get("lith") or u.get("name_long") or ""
            if isinstance(lith, list) and lith:
                lith = lith[0].get("name", "") if isinstance(lith[0], dict) \
                       else str(lith[0])
            layers.append((classify_lith(str(lith)), t))
    except Exception:
        pass
    if not layers:
        layers = fallback
    total = sum(t for _, t in layers)
    col, z = [], 0.0
    for cls, t in layers:
        dz = t / total
        col.append((z, z + dz, cls))
        z += dz
    return col


def lith_at(x, z, column):
    if not torch.is_tensor(z):
        z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    z = z.to(DEVICE); x = x.to(DEVICE)
    z_eff = (z + DIP_AMP * torch.sin(math.pi * x / AR)).clamp(0.0, 0.999)
    out = torch.full(z.shape, N_LITH - 1, dtype=torch.long, device=DEVICE)
    for top, bot, cls in column:
        m = (z_eff >= top) & (z_eff < bot)
        out = torch.where(m,
                          torch.tensor(cls, dtype=torch.long, device=DEVICE),
                          out)
    return out


# ---------------------------------------------------------- mines & validation
def project_mines(data_dir, site_lat, site_lon):
    pts = []
    try:
        data = json.loads((data_dir / "mines.json").read_text())
        for e in data.get("elements", []):
            lat = e.get("lat") or (e.get("center") or {}).get("lat")
            lon = e.get("lon") or (e.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            dx = (lon - site_lon) * 111000.0 * math.cos(math.radians(site_lat))
            dy = (lat - site_lat) * 111000.0
            d = math.hypot(dx, dy)
            if d > LX_M:
                continue
            pts.append(AR * d / LX_M)
    except Exception:
        pass
    if not pts:
        pts = [2.0, 5.0, 8.0]
    return pts


def split_mines(mines, val_frac, seed=0):
    if val_frac <= 0 or len(mines) < 5:
        return list(mines), []
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(mines), generator=g).tolist()
    n_val = max(1, int(round(len(mines) * val_frac)))
    val_set = set(idx[:n_val])
    train = [m for i, m in enumerate(mines) if i not in val_set]
    val   = [mines[i] for i in sorted(val_set)]
    return train, val


# ---------------------------------------------------------------- heat flow
def load_heat_flow_target(data_dir, site_key, fallback_mwm2):
    p = data_dir / "heat_flow.json"
    hf = fallback_mwm2
    if p.exists():
        try:
            d = json.loads(p.read_text())
            hf = float(d.get("hf_mwm2", fallback_mwm2))
        except Exception:
            pass
    T_target = hf / HF_REF_MWM2
    T_target = max(0.4, min(1.4, T_target))
    return float(T_target), float(hf)


def _xline_kde(pts_x, n_grid=X_GRID_N, bw=BW_X,
               floor=KMOD_FLOOR, gain=KMOD_GAIN):
    grid = torch.linspace(0, AR, n_grid, device=DEVICE)
    if not pts_x:
        return grid, torch.ones(n_grid, device=DEVICE)
    pts = torch.tensor(pts_x, dtype=torch.float32, device=DEVICE)
    K = torch.exp(-0.5 * ((grid[:, None] - pts[None, :]) / bw) ** 2).sum(dim=1)
    K = K / K.mean().clamp(min=1e-6)
    return grid, floor + gain * K


def _earthquake_xpts(data_dir, site_lat, site_lon):
    p = data_dir / "earthquakes.json"
    pts = []
    if not p.exists():
        return pts
    try:
        d = json.loads(p.read_text())
        for f in d.get("features", []):
            geom = f.get("geometry", {}) or {}
            coords = geom.get("coordinates", []) or []
            if len(coords) < 2:
                continue
            lon_e, lat_e = float(coords[0]), float(coords[1])
            dx = (lon_e - site_lon) * 111000.0 \
                 * math.cos(math.radians(site_lat))
            dy = (lat_e - site_lat) * 111000.0
            dm = math.hypot(dx, dy)
            if dm > LX_M:
                continue
            pts.append(AR * dm / LX_M)
    except Exception:
        pass
    return pts


def _fault_xpts(data_dir, site_lat, site_lon):
    p = data_dir / "faults.geojson"
    if not p.exists():
        p = data_dir / "faults.json"
    pts = []
    if not p.exists():
        return pts
    try:
        d = json.loads(p.read_text())
        for f in d.get("features", []):
            geom = f.get("geometry", {}) or {}
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", []) or []
            if gtype == "LineString":
                lines = [coords]
            elif gtype == "MultiLineString":
                lines = coords
            else:
                continue
            for line in lines:
                for v in line:
                    if len(v) < 2:
                        continue
                    lon_e, lat_e = float(v[0]), float(v[1])
                    dx = (lon_e - site_lon) * 111000.0 \
                         * math.cos(math.radians(site_lat))
                    dy = (lat_e - site_lat) * 111000.0
                    dm = math.hypot(dx, dy)
                    if dm > LX_M:
                        continue
                    pts.append(AR * dm / LX_M)
    except Exception:
        pass
    return pts


def find_contacts_robust(column, type_a, type_b):
    contacts = []
    n = len(column)
    pair = {type_a, type_b}
    for i in range(n):
        _, b_i, c_i = column[i]
        if c_i == OTH:
            continue
        j = i + 1
        while j < n and column[j][2] == OTH:
            j += 1
        if j >= n:
            continue
        t_j, _, c_j = column[j]
        if {c_i, c_j} == pair:
            contacts.append(0.5 * (b_i + t_j))
    return contacts


def _bumps_from_contacts(contacts, n_grid=Z_GRID_N, bw=BW_Z,
                         floor=SRC_FLOOR, gain=SRC_GAIN):
    grid = torch.linspace(0.0, 1.0, n_grid, device=DEVICE)
    if not contacts:
        return grid, torch.ones(n_grid, device=DEVICE)
    cz = torch.tensor(contacts, dtype=torch.float32, device=DEVICE)
    K = torch.exp(-0.5 * ((grid[:, None] - cz[None, :]) / bw) ** 2).sum(dim=1)
    K = K / K.max().clamp(min=1e-6)
    return grid, floor + gain * K


def _deep_intrusive_bump(column, n_grid=Z_GRID_N, bw=BW_Z * 2.0,
                         floor=SRC_FLOOR, gain=SRC_GAIN_DEEP):
    grid = torch.linspace(0.0, 1.0, n_grid, device=DEVICE)
    deep_zs = [0.5 * (t + b) for t, b, c in column if c == INT]
    if not deep_zs:
        return grid, torch.ones(n_grid, device=DEVICE), None
    z_deep = max(deep_zs)
    K = torch.exp(-0.5 * ((grid - z_deep) / bw) ** 2)
    return grid, floor + gain * K, z_deep


PAIR_CASCADE = {
    "magmatic_sulfide": [(INT, EVA, "intrusive×evaporite"),
                         (BAS, INT, "basalt×intrusive"),
                         (SED, INT, "sediment×intrusive")],
    "sediment_cu":     [(SED, EVA, "sediment×evaporite"),
                         (SED, BAS, "sediment×basalt"),
                         (SED, INT, "sediment×intrusive")],
    "orogenic_au":     [(SED, INT, "sediment×intrusive"),
                         (BAS, INT, "basalt×intrusive"),
                         (SED, BAS, "sediment×basalt")],
    "hydrothermal":    [(SED, BAS, "sediment×basalt"),
                         (SED, INT, "sediment×intrusive")],
}


def build_proxy_modulators(data_dir, site_lat, site_lon,
                           deposit_type, column):
    k_mod_x = torch.ones(X_GRID_N, device=DEVICE)
    src_z   = torch.ones(Z_GRID_N, device=DEVICE)
    meta = {"deposit_type": deposit_type,
            "proxies_used": [], "proxy_chain": [],
            "n_fault_pts": 0, "n_earthquakes": 0,
            "n_contacts": 0, "contact_pair": None,
            "k_mod_range": [1.0, 1.0], "src_z_range": [1.0, 1.0]}

    fpts = _fault_xpts(data_dir, site_lat, site_lon)
    epts = _earthquake_xpts(data_dir, site_lat, site_lon)
    meta["n_fault_pts"]   = len(fpts)
    meta["n_earthquakes"] = len(epts)

    if fpts:
        gain = 2.0 * KMOD_GAIN if deposit_type == "orogenic_au" else KMOD_GAIN
        _, K = _xline_kde(fpts, gain=gain)
        k_mod_x = K
        meta["proxies_used"].append("faults")
        meta["proxy_chain"].append(f"k_mod_x: faults({len(fpts)} pts)")
    elif epts and deposit_type in ("magmatic_sulfide", "orogenic_au",
                                   "kimberlite", "hydrothermal"):
        _, K = _xline_kde(epts, gain=KMOD_GAIN_EQ)
        k_mod_x = K
        meta["proxies_used"].append("seismicity_fallback")
        meta["proxy_chain"].append(
            f"k_mod_x: seismicity({len(epts)} pts) [fault fallback]")
    else:
        meta["proxy_chain"].append("k_mod_x: none (uniform)")

    if deposit_type == "kimberlite":
        _, S, z_deep = _deep_intrusive_bump(column)
        if z_deep is not None:
            src_z = S
            meta["proxies_used"].append("deep_intrusive_root")
            meta["proxy_chain"].append(
                f"src_z: deepest intrusive @ z={z_deep:.2f} (root zone)")
            meta["contact_pair"] = "deep_intrusive_root"
        else:
            meta["proxy_chain"].append("src_z: no intrusive layer present")
    elif deposit_type in PAIR_CASCADE:
        fired = False
        for a, b, name in PAIR_CASCADE[deposit_type]:
            cs = find_contacts_robust(column, a, b)
            if cs:
                _, S = _bumps_from_contacts(cs)
                src_z = S
                meta["proxies_used"].append(f"contacts:{name}")
                meta["proxy_chain"].append(
                    f"src_z: {name} ({len(cs)} contacts)")
                meta["n_contacts"]   = len(cs)
                meta["contact_pair"] = name
                fired = True
                break
        if not fired:
            meta["proxy_chain"].append("src_z: none (no matching contacts)")
    else:
        meta["proxy_chain"].append(f"src_z: unknown deposit_type {deposit_type}")

    if not meta["proxies_used"]:
        meta["proxies_used"].append("none(all_modulators=1)")

    meta["k_mod_range"] = [float(k_mod_x.min().item()),
                           float(k_mod_x.max().item())]
    meta["src_z_range"] = [float(src_z.min().item()),
                           float(src_z.max().item())]
    return k_mod_x, src_z, meta


# ---------------------------- NEW: auto z-prior derivation -------------------
def auto_z_prior_from_src(src_z_grid, src_z, threshold=AUTO_ZPRIOR_THRESH):
    """Return argmax-z of src_z if the proxy is non-flat, else None.

    Threshold is on max(src_z) — uniform src_z = SRC_FLOOR = 1.0; an active
    contact bump rises to ~SRC_FLOOR + SRC_GAIN ~ 4.0. We accept anything
    above 1.5 as 'meaningfully bumped'.
    """
    if not torch.is_tensor(src_z):
        src_z = torch.as_tensor(src_z, dtype=torch.float32, device=DEVICE)
    if float(src_z.max().item()) < threshold:
        return None
    iz = int(src_z.argmax().item())
    return float(src_z_grid[iz].item())


# ---------------------------------------------------------------- networks
class MLP(nn.Module):
    def __init__(self, h=64, depth=5, out_bias=0.0):
        super().__init__()
        L = [nn.Linear(2, h), nn.Tanh()]
        for _ in range(depth - 1):
            L += [nn.Linear(h, h), nn.Tanh()]
        last = nn.Linear(h, 1)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, out_bias)
        L += [last]
        self.net = nn.Sequential(*L)

    def forward(self, xz):
        return self.net(xz).squeeze(-1)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fT = MLP(out_bias=0.0)
        self.fP = MLP(out_bias=0.0)
        self.fC = MLP(out_bias=0.5)

        self.log_k = nn.Parameter(torch.tensor([-1.0, -1.5, -3.0, -3.5, -2.0]))
        self.log_Ceq_lith = nn.Parameter(
            torch.tensor([-1.5, -1.2, -1.0, -3.5, -1.5]))
        self.b_T   = nn.Parameter(torch.tensor(1.0))
        self.log_kr = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("T_bottom", torch.tensor(1.0))
        self.register_buffer("k_mod_x_grid", torch.linspace(0, AR, X_GRID_N))
        self.register_buffer("k_mod_x",      torch.ones(X_GRID_N))
        self.register_buffer("src_z_grid",   torch.linspace(0, 1, Z_GRID_N))
        self.register_buffer("src_z",        torch.ones(Z_GRID_N))

    def attach_site_data(self, T_bottom, k_mod_x, src_z):
        with torch.no_grad():
            self.T_bottom.fill_(float(T_bottom))
            self.k_mod_x.copy_(k_mod_x.to(self.k_mod_x.device))
            self.src_z.copy_(src_z.to(self.src_z.device))

    @staticmethod
    def _interp1d(grid, vals, q):
        n = grid.numel()
        if n < 2:
            return torch.ones_like(q)
        x_lo = grid[0]; x_hi = grid[-1]
        f = (q - x_lo) / (x_hi - x_lo).clamp(min=1e-9)
        f = f.clamp(0.0, 1.0) * (n - 1)
        i0 = f.long().clamp(0, n - 2)
        t  = f - i0.float()
        return vals[i0] * (1 - t) + vals[i0 + 1] * t

    def k_mod(self, x):
        return self._interp1d(self.k_mod_x_grid, self.k_mod_x, x)

    def src_mod(self, z):
        return self._interp1d(self.src_z_grid, self.src_z, z)

    def fields(self, x, z):
        xz = torch.stack([x / AR, z], dim=-1)
        T = torch.sigmoid(self.fT(xz))
        P = self.fP(xz)
        C = F.softplus(self.fC(xz))
        return T, P, C


def grad(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def residuals(model, x_in, z_in, lith, intr):
    x = x_in.detach().clone().requires_grad_(True)
    z = z_in.detach().clone().requires_grad_(True)
    T, P, C = model.fields(x, z)
    Tx, Tz = grad(T, x), grad(T, z)
    Px, Pz = grad(P, x), grad(P, z)
    Cx, Cz = grad(C, x), grad(C, z)
    Txx, Tzz = grad(Tx, x), grad(Tz, z)
    Cxx, Czz = grad(Cx, x), grad(Cz, z)

    k_lith = torch.exp(model.log_k[lith])
    k_eff  = k_lith * model.k_mod(x)
    qx = -k_eff * Px
    qz = -k_eff * (Pz - RA * T)
    Rd = grad(qx, x) + grad(qz, z)
    Q  = 0.5 * intr
    Rh = qx * Tx + qz * Tz - (Txx + Tzz) / PE_T - Q

    log_Ceq = model.log_Ceq_lith[lith] + model.b_T * T
    sigma   = torch.log(C + 1e-3) - log_Ceq
    kr_eff  = torch.exp(model.log_kr) * model.src_mod(z)
    Rrate   = kr_eff * F.softplus(sigma, beta=2.0)

    Rc = qx * Cx + qz * Cz - (Cxx + Czz) / PE_C + Rrate
    return Rd, Rh, Rc, Rrate


def boundary_loss(model, column, n=300):
    xb = torch.rand(n, device=DEVICE) * AR
    zt = torch.zeros(n, device=DEVICE)
    Tt, Pt, _  = model.fields(xb, zt)
    Ltop = (Tt**2).mean() + (Pt**2).mean()

    zb_q = torch.full((n,), 0.999, device=DEVICE)
    lith_b = lith_at(xb, zb_q, column)
    src = ((lith_b == BAS) | (lith_b == INT)).float()
    C_target = 0.2 + 0.8 * src
    zb = torch.ones(n, device=DEVICE)
    Tb, _, Cb = model.fields(xb, zb)
    Lbot = ((Tb - model.T_bottom)**2).mean() + ((Cb - C_target)**2).mean()
    return Ltop + Lbot


TARGET_M  = 0.7
MARGIN    = 0.20
RARE_CAP  = 0.50
W_PHYS    = 1.0
W_BND     = 5.0
W_POS     = 5.0
W_CON     = 8.0
W_RARE    = 0.10


def mine_M(model, mx, column, n_z=24):
    n_m = mx.shape[0]
    if n_m == 0:
        zero = torch.zeros(0, device=DEVICE)
        return zero, zero
    z_grid = torch.linspace(0.05, 0.95, n_z, device=DEVICE)
    xx = mx.unsqueeze(1).expand(-1, n_z).reshape(-1)
    zz = z_grid.unsqueeze(0).expand(n_m, -1).reshape(-1)
    lith = lith_at(xx, zz, column)
    intr = (lith == INT).float()
    _, _, _, Rr = residuals(model, xx, zz, lith, intr)
    Rr = Rr.view(n_m, n_z)
    w = torch.softmax(8.0 * Rr, dim=1)
    M_smooth = (w * Rr).sum(dim=1)
    M_max    = Rr.max(dim=1).values
    return M_smooth, M_max


def pu_losses(model, x, z, lith, intr, mx, column):
    _, _, _, Rrate_bg = residuals(model, x, z, lith, intr)
    M_mine_smooth, M_mine_max = mine_M(model, mx, column)
    Lpos  = -torch.log(M_mine_smooth + 1e-3).mean() \
            + (M_mine_smooth - TARGET_M).pow(2).mean()
    Lrare = torch.relu(Rrate_bg - RARE_CAP).pow(2).mean()
    Lcon  = torch.relu(MARGIN + Rrate_bg.mean() - M_mine_smooth.mean())
    return Lpos, Lrare, Lcon, Rrate_bg, M_mine_max


def z_peak_soft(model, column, n_z=24):
    zg = torch.linspace(0.05, 0.95, n_z, device=DEVICE)
    xg = torch.full_like(zg, AR / 2)
    lg = lith_at(xg, zg, column); ig = (lg == INT).float()
    _, _, _, Rzg = residuals(model, xg, zg, lg, ig)
    w  = torch.softmax(8.0 * Rzg, dim=0)
    return (w * zg).sum(), Rzg, zg


def compute_M(model, x, z, column):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    if not torch.is_tensor(z):
        z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
    lith = lith_at(x, z, column)
    intr = (lith == INT).float()
    _, _, _, Rrate = residuals(model, x, z, lith, intr)
    return Rrate.detach()


def mine_M_eval(model, mines, column):
    if len(mines) == 0:
        return float("nan")
    mx = torch.tensor(mines, dtype=torch.float32, device=DEVICE)
    _, M_max = mine_M(model, mx, column)
    return float(M_max.mean().item())


def sensitivity_probe(model, column, mines, deltas=(-0.5, 0.5)):
    base = mine_M_eval(model, mines, column)
    if base <= 0 or math.isnan(base):
        return {"base": base, "note": "no mines / zero base"}
    out = {"base_M_at_mines": base, "params": {}, "scalars": {}}

    for pname in ("log_k", "log_Ceq_lith"):
        param = getattr(model, pname)
        per_lith = {}
        for i, lname in enumerate(LITH_NAMES):
            shifts = []
            for d in deltas:
                old = param.data[i].item()
                param.data[i] = old + d
                new = mine_M_eval(model, mines, column)
                param.data[i] = old
                shifts.append({"delta": d, "M_ratio": new / base})
            per_lith[lname] = shifts
        out["params"][pname] = per_lith

    bT_shifts = []
    for d in deltas:
        old = model.b_T.data.item()
        model.b_T.data = torch.tensor(old + d, device=model.b_T.device)
        new = mine_M_eval(model, mines, column)
        model.b_T.data = torch.tensor(old, device=model.b_T.device)
        bT_shifts.append({"delta": d, "M_ratio": new / base})
    out["scalars"]["b_T"] = bT_shifts
    return out


def proxy_ablation(model, column, mines):
    base = mine_M_eval(model, mines, column)
    out = {"base_M_at_mines": base}
    if base <= 0 or math.isnan(base):
        out["note"] = "no mines / zero base"
        return out
    k_save = model.k_mod_x.detach().clone()
    s_save = model.src_z.detach().clone()
    with torch.no_grad():
        model.k_mod_x.fill_(1.0)
    out["k_mod_off_M_ratio"] = mine_M_eval(model, mines, column) / base
    with torch.no_grad():
        model.k_mod_x.copy_(k_save); model.src_z.fill_(1.0)
    out["src_z_off_M_ratio"] = mine_M_eval(model, mines, column) / base
    with torch.no_grad():
        model.k_mod_x.fill_(1.0)
    out["both_off_M_ratio"] = mine_M_eval(model, mines, column) / base
    with torch.no_grad():
        model.k_mod_x.copy_(k_save); model.src_z.copy_(s_save)
    return out


def train_one(site_key, seed, epochs, n_col, lr,
              mines_train, mines_val, column,
              z_prior, w_zprior,
              T_bottom, k_mod_x, src_z, verbose=True):
    torch.manual_seed(seed)
    model = PINN().to(DEVICE)
    model.attach_site_data(T_bottom=T_bottom, k_mod_x=k_mod_x, src_z=src_z)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    mx_tr = torch.tensor(mines_train, dtype=torch.float32, device=DEVICE)

    for ep in range(epochs):
        x = torch.rand(n_col, device=DEVICE) * AR
        z = torch.rand(n_col, device=DEVICE)
        lith = lith_at(x, z, column)
        intr = (lith == INT).float()
        Rd, Rh, Rc, _ = residuals(model, x, z, lith, intr)
        Lphys = (Rd**2).mean() + (Rh**2).mean() + (Rc**2).mean()
        Lbnd  = boundary_loss(model, column)
        Lpos, Lrare, Lcon, Rrate, Mm = pu_losses(model, x, z, lith, intr,
                                                 mx_tr, column)

        loss = (W_PHYS*Lphys + W_BND*Lbnd
                + W_POS*Lpos + W_CON*Lcon + W_RARE*Lrare)

        Lz_val = 0.0
        if z_prior is not None and w_zprior > 0.0:
            zpk, _, _ = z_peak_soft(model, column)
            Lz = (zpk - z_prior).pow(2)
            loss = loss + w_zprior * Lz
            Lz_val = float(Lz.item())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if verbose and ep % 200 == 0:
            zpk, Rzg, zg = z_peak_soft(model, column)
            z_argmax = float(zg[Rzg.argmax()].item())
            mval = mine_M_eval(model, mines_val, column) \
                   if mines_val else float("nan")
            extra = f" zprior={Lz_val:.3f}" if z_prior is not None else ""
            print(f"  seed={seed} ep {ep:4d} | L={loss.item():.3f} "
                  f"phys={Lphys.item():.3f} bnd={Lbnd.item():.3f} "
                  f"pos={Lpos.item():.3f} con={Lcon.item():.3f} "
                  f"rare={Lrare.item():.3f}{extra} "
                  f"M@tr={Mm.mean().item():.2f} M@val={mval:.2f} "
                  f"M_bg={Rrate.mean().item():.2f} "
                  f"M_max={Rrate.max().item():.2f} z*={z_argmax:.2f} "
                  f"b_T={model.b_T.item():.2f}")

    train_score = mine_M_eval(model, mines_train, column)
    val_score   = mine_M_eval(model, mines_val, column) \
                  if mines_val else train_score
    return model, train_score, val_score


def train(site_key, epochs=2000, n_col=2000, lr=2e-3,
          val_frac=0.2, seeds=1, z_prior_cli=None, w_zprior=0.5):
    s = SITES[site_key]
    data_dir = Path("data") / site_key
    out_dir  = Path("output") / site_key
    out_dir.mkdir(parents=True, exist_ok=True)
    column = build_column(data_dir, s["fallback_column"])
    mines  = project_mines(data_dir, s["lat"], s["lon"])

    z_prior = z_prior_cli if z_prior_cli is not None else s.get("z_prior")
    w_zp    = s.get("z_prior_w", w_zprior)

    deposit_type = s.get("deposit_type") \
                   or DEPOSIT_TYPE_BY_SITE.get(site_key, DEFAULT_DEPOSIT_TYPE)

    T_bottom, hf_mwm2 = load_heat_flow_target(
        data_dir, site_key, s.get("heat_flow_mwm2", 55.0))

    k_mod_x, src_z, proxy_meta = build_proxy_modulators(
        data_dir, s["lat"], s["lon"], deposit_type, column)

    # ---------- NEW: auto z-prior fallback from proxy src_z ----------
    z_prior_source = "site/cli" if z_prior is not None else None
    if z_prior is None:
        srcz_grid_t = torch.linspace(0, 1, src_z.numel(), device=DEVICE)
        z_auto = auto_z_prior_from_src(srcz_grid_t, src_z)
        if z_auto is not None:
            z_prior = z_auto
            z_prior_source = "auto(src_z argmax)"

    print(f"\n=== Training: {s['name']} ===")
    print(f"Deposit type: {deposit_type}")
    print(f"Proxies used: {proxy_meta['proxies_used']}")
    for line in proxy_meta["proxy_chain"]:
        print(f"  {line}")
    if proxy_meta["n_fault_pts"]:
        print(f"  fault vertices projected: {proxy_meta['n_fault_pts']}")
    if proxy_meta["n_earthquakes"]:
        print(f"  earthquakes projected:    {proxy_meta['n_earthquakes']}")
    if proxy_meta["n_contacts"]:
        print(f"  contacts ({proxy_meta['contact_pair']}): "
              f"{proxy_meta['n_contacts']}")
    print(f"  k_mod_x range: [{proxy_meta['k_mod_range'][0]:.2f}, "
          f"{proxy_meta['k_mod_range'][1]:.2f}]")
    print(f"  src_z   range: [{proxy_meta['src_z_range'][0]:.2f}, "
          f"{proxy_meta['src_z_range'][1]:.2f}]")
    print("Column:")
    for c in column:
        print(f"  {c[0]:.3f}  {c[1]:.3f}  {LITH_NAMES[c[2]]}")
    print(f"Projected mines: {len(mines)}")
    print(f"Heat flow:  {hf_mwm2:.1f} mW/m^2  -> bottom T_target={T_bottom:.2f}")
    if z_prior is not None:
        print(f"z_prior = {z_prior:.2f}  (w={w_zp})  source={z_prior_source}")
    else:
        print("z_prior = none")

    mines_tr, mines_val = split_mines(mines, val_frac, seed=0)
    print(f"split: train={len(mines_tr)}  val={len(mines_val)}")

    best = None
    for seed in range(max(1, seeds)):
        print(f"--- seed {seed}/{seeds-1} ---")
        model, tr, vl = train_one(site_key, seed, epochs, n_col, lr,
                                  mines_tr, mines_val, column,
                                  z_prior, w_zp,
                                  T_bottom, k_mod_x, src_z)
        print(f"  -> seed={seed} train_M={tr:.3f} val_M={vl:.3f}")
        score = vl if mines_val else tr
        if best is None or score > best["score"]:
            best = {"seed": seed, "model": model,
                    "train_M": tr, "val_M": vl, "score": score}

    model = best["model"]
    print(f"\nselected seed={best['seed']} "
          f"train_M={best['train_M']:.3f} val_M={best['val_M']:.3f}")

    print("running sensitivity probe ...")
    sens = sensitivity_probe(model, column, mines)
    sens["site_inputs"] = {
        "deposit_type": deposit_type,
        "heat_flow_mwm2": hf_mwm2,
        "T_bottom": T_bottom,
        "proxy_meta": proxy_meta,
        "z_prior": z_prior,
        "z_prior_source": z_prior_source,
    }

    print("running proxy ablation ...")
    abl = proxy_ablation(model, column, mines)
    sens["proxy_ablation"] = abl
    if "k_mod_off_M_ratio" in abl:
        print(f"  ablate k_mod_x  -> M_ratio={abl['k_mod_off_M_ratio']:.3f}")
        print(f"  ablate src_z    -> M_ratio={abl['src_z_off_M_ratio']:.3f}")
        print(f"  ablate both     -> M_ratio={abl['both_off_M_ratio']:.3f}")

    (out_dir / "sensitivity.json").write_text(json.dumps(sens, indent=2))

    if "params" in sens:
        worst = []
        for pname, per_lith in sens["params"].items():
            for lname, shifts in per_lith.items():
                for sh in shifts:
                    worst.append((abs(1 - sh["M_ratio"]),
                                  pname, lname, sh["delta"], sh["M_ratio"]))
        for sh in sens.get("scalars", {}).get("b_T", []):
            worst.append((abs(1 - sh["M_ratio"]),
                          "b_T", "scalar", sh["delta"], sh["M_ratio"]))
        worst.sort(reverse=True)
        print("top 5 most-sensitive perturbations:")
        for w in worst[:5]:
            print(f"  {w[1]}[{w[2]}] {w[3]:+.2f}  -> M_ratio={w[4]:.3f}")

    torch.save({"model_state": model.state_dict(),
                "column": column,
                "mines": mines,
                "mines_train": mines_tr,
                "mines_val": mines_val,
                "site": site_key,
                "deposit_type": deposit_type,
                "selected_seed": best["seed"],
                "train_M": best["train_M"],
                "val_M": best["val_M"],
                "heat_flow_mwm2": hf_mwm2,
                "T_bottom": T_bottom,
                "z_prior": z_prior,
                "z_prior_source": z_prior_source,
                "proxy_meta": proxy_meta},
               out_dir / "model.pt")
    print(f"saved {out_dir/'model.pt'}")
    print(f"saved {out_dir/'sensitivity.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT, choices=list(SITES.keys()))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--z_prior", type=float, default=None)
    ap.add_argument("--zprior_w", type=float, default=0.5)
    a = ap.parse_args()
    train(a.site, epochs=a.epochs, seeds=a.seeds,
          val_frac=a.val_frac, z_prior_cli=a.z_prior, w_zprior=a.zprior_w)