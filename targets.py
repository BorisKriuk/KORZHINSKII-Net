"""targets.py — multi-site geographic drill-target map (v2).

Updated for pinn.py v2:
  - lith_at(x, z, column)
  - mines is List[float] (x only)
  - uses compute_M()
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects as pe
from pinn import PINN, compute_M, AR, LX_M, DEVICE
from sites import SITES, DEFAULT

NX, NZ            = 400, 200
TOP_K             = 8
MIN_SEP_KM        = 6.0
EXCL_MINE_KM      = 3.0
Z_MIN, Z_MAX      = 0.04, 0.96
PEAK_KERNEL       = 5
PEAK_TOP_N        = 120
SCORE_FLOOR_FRAC  = 0.10
SCORE_FLOOR_ABS   = 1e-5
SIGMA_AZ_DEG      = 22.0
AZ_BONUS          = 1.6
GRID_N            = 280
COL_THICKNESS_M   = 5000.0


def haversine_km(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def latlon_from_polar(r_km, az, lat0, lon0):
    cl = math.cos(math.radians(lat0))
    return (lat0 + r_km * math.cos(az) / 111.0,
            lon0 + r_km * math.sin(az) / (111.0 * cl))


def local_maxima_mask(arr, k=2):
    m = np.ones_like(arr, dtype=bool)
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            if di == 0 and dj == 0:
                continue
            sh = np.roll(np.roll(arr, di, 0), dj, 1)
            m &= arr >= sh
    m[:k, :] = False; m[-k:, :] = False
    m[:, :k] = False; m[:, -k:] = False
    return m


def load_known_mines(data_dir):
    out = []
    p = data_dir / "mines.json"
    if not p.exists():
        return out
    try:
        d = json.loads(p.read_text())
        for e in d.get("elements", []):
            lat = e.get("lat") or (e.get("center") or {}).get("lat")
            lon = e.get("lon") or (e.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            t = e.get("tags", {})
            name = t.get("name") or t.get("man_made") \
                   or t.get("industrial") or "mine"
            out.append({"lat": float(lat), "lon": float(lon),
                        "name": str(name)})
    except Exception:
        pass
    return out


def load_towns(data_dir):
    p = data_dir / "towns.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def merge_landmarks(static, fetched):
    out = list(static)
    for f in fetched:
        if any(haversine_km(f["lat"], f["lon"],
                            sl["lat"], sl["lon"]) < 1.5 for sl in out):
            continue
        out.append(f)
    return out


def main(site_key):
    s = SITES[site_key]
    SITE_LAT, SITE_LON = s["lat"], s["lon"]
    RADIUS_KM = s["radius_km"]
    data_dir = Path("data") / site_key
    out_dir  = Path("output") / site_key
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(out_dir / "model.pt",
                      map_location=DEVICE, weights_only=False)
    model = PINN().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    column = ckpt["column"]

    # ---------- evaluate M on (x, z) grid ----------
    xs = torch.linspace(0, AR, NX, device=DEVICE)
    zs = torch.linspace(0, 1,  NZ, device=DEVICE)
    X, Z = torch.meshgrid(xs, zs, indexing="xy")
    x = X.flatten(); z = Z.flatten()
    M = compute_M(model, x, z, column).cpu().numpy().reshape(NZ, NX)

    z_axis = zs.cpu().numpy()
    x_axis = xs.cpu().numpy()
    r_km_axis = x_axis * (LX_M / AR) / 1000.0
    z_mask = (z_axis >= Z_MIN) & (z_axis <= Z_MAX)
    M_dm = np.where(z_mask[:, None], M, -np.inf)

    M_finite_max = float(M_dm[np.isfinite(M_dm)].max()) \
                   if np.isfinite(M_dm).any() else 0.0
    score_floor  = max(SCORE_FLOOR_ABS, SCORE_FLOOR_FRAC * M_finite_max)
    print(f"  M_max={M_finite_max:.4f}   score_floor={score_floor:.4g}")

    # ---------- peaks ----------
    finite = np.where(np.isfinite(M_dm), M_dm, M_dm.min())
    lmax = local_maxima_mask(finite, k=PEAK_KERNEL // 2) & np.isfinite(M_dm)
    iz, ix = np.where(lmax)
    if len(iz) == 0:
        iz_top = np.argmax(np.where(np.isfinite(M_dm), M_dm, -np.inf), axis=0)
        ix = np.arange(NX)
        iz = iz_top
    sc = M_dm[iz, ix]
    keep = np.isfinite(sc) & (sc > score_floor)
    iz, ix, sc = iz[keep], ix[keep], sc[keep]
    if len(sc) == 0:
        flat = np.where(np.isfinite(M_dm), M_dm, -np.inf)
        izg, ixg = np.unravel_index(np.argmax(flat), flat.shape)
        iz, ix, sc = (np.array([izg]), np.array([ixg]),
                      np.array([flat[izg, ixg]]))
    o = np.argsort(-sc)[:PEAK_TOP_N]
    peaks = [(float(x_axis[ix[k]]), float(z_axis[iz[k]]), float(sc[k]))
             for k in o]
    print(f"  {len(peaks)} candidate peaks")

    # ---------- known mines / azimuths ----------
    known = load_known_mines(data_dir)
    coslat = math.cos(math.radians(SITE_LAT))
    mine_az = []
    for m in known:
        dy = (m["lat"] - SITE_LAT) * 111.0
        dx = (m["lon"] - SITE_LON) * 111.0 * coslat
        if math.hypot(dx, dy) > RADIUS_KM:
            continue
        mine_az.append(math.atan2(dx, dy))
    sigma = math.radians(SIGMA_AZ_DEG)

    # ---------- pick targets ----------
    AZS = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    picked = []
    for x_n, z_n, sv in peaks:
        r_km = x_n * (LX_M / AR) / 1000.0
        if r_km < 3.0 or r_km > RADIUS_KM:
            continue
        depth_m = z_n * COL_THICKNESS_M
        best = None
        for az in AZS:
            lat, lon = latlon_from_polar(r_km, az, SITE_LAT, SITE_LON)
            if any(haversine_km(lat, lon, p["lat"], p["lon"]) < MIN_SEP_KM
                   for p in picked):
                continue
            if any(haversine_km(lat, lon, m["lat"], m["lon"]) < EXCL_MINE_KM
                   for m in known):
                continue
            if mine_az:
                d = (np.array(mine_az) - az + np.pi) % (2 * np.pi) - np.pi
                w = np.exp(-(d**2) / (2 * sigma**2)).sum() / len(mine_az)
                mod = 1.0 + AZ_BONUS * w
            else:
                mod = 1.0
            sg = sv * mod
            if best is None or sg > best["score"]:
                best = {"lat": lat, "lon": lon, "score": sg,
                        "dist_km": r_km, "depth_m": depth_m,
                        "az_deg": (math.degrees(az) + 360) % 360}
        if best is not None:
            best["rank"] = len(picked) + 1
            picked.append(best)
        if len(picked) >= TOP_K:
            break

    # ---------- raster heatmap ----------
    half_lat = RADIUS_KM / 111.0
    half_lon = RADIUS_KM / (111.0 * coslat)
    lats = np.linspace(SITE_LAT - half_lat, SITE_LAT + half_lat, GRID_N)
    lons = np.linspace(SITE_LON - half_lon, SITE_LON + half_lon, GRID_N)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    DY = (LAT - SITE_LAT) * 111.0
    DX = (LON - SITE_LON) * 111.0 * coslat
    R_grid  = np.sqrt(DX * DX + DY * DY)
    AZ_grid = np.arctan2(DX, DY)

    M_r = M_dm.max(axis=0)
    M_r = np.where(np.isfinite(M_r), M_r, 0.0)
    M_base = np.interp(R_grid.ravel(), r_km_axis, M_r,
                       left=M_r[0], right=0.0).reshape(R_grid.shape)
    if mine_az:
        d = (AZ_grid[..., None] - np.array(mine_az)[None, None, :] + np.pi) \
            % (2 * np.pi) - np.pi
        w = np.exp(-(d**2) / (2 * sigma**2)).sum(axis=-1) / len(mine_az)
        modul = 1.0 + AZ_BONUS * w
    else:
        modul = np.ones_like(R_grid)
    M_geo = np.where(R_grid <= RADIUS_KM, M_base * modul, np.nan)

    # ---------- landmarks ----------
    landmarks = merge_landmarks(s.get("landmarks", []),
                                load_towns(data_dir))
    landmarks = [l for l in landmarks
                 if haversine_km(l["lat"], l["lon"],
                                 SITE_LAT, SITE_LON) <= RADIUS_KM]
    print(f"  {len(landmarks)} landmarks within {RADIUS_KM:.0f} km")

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(13, 11), facecolor="#0b0d12")
    ax.set_facecolor("#0b0d12")
    cmap = LinearSegmentedColormap.from_list("p",
        [(0, "#0b0d12"), (.15, "#1a1340"), (.40, "#5a1a8a"),
         (.62, "#d83a87"), (.82, "#ff8a3a"), (1, "#fff066")])
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    im = ax.imshow(M_geo, extent=extent, origin="lower", cmap=cmap,
                   aspect=1.0 / coslat, interpolation="bilinear",
                   alpha=0.92)

    th = np.linspace(0, 2 * np.pi, 240)
    for r_ring, ls, alp in [(RADIUS_KM, "--", .5),
                             (RADIUS_KM * .5, ":", .3),
                             (RADIUS_KM * .2, ":", .22)]:
        rl  = SITE_LAT + (r_ring / 111.0) * np.cos(th)
        rln = SITE_LON + (r_ring / (111.0 * coslat)) * np.sin(th)
        ax.plot(rln, rl, ls, color="#5af0a0", lw=.9, alpha=alp)

    for m in known:
        ax.plot(m["lon"], m["lat"], "x", color="#ff5b6b",
                ms=7, mew=1.8, zorder=5)
    if known:
        ax.plot([], [], "x", color="#ff5b6b", ms=7, mew=1.8,
                label=f"OSM mine ({len(known)})")

    for m in s.get("district_mines", []):
        if haversine_km(m["lat"], m["lon"], SITE_LAT, SITE_LON) > RADIUS_KM:
            continue
        ax.plot(m["lon"], m["lat"], "s", mec="#ff8e6e", mfc="none",
                ms=8, mew=1.6, zorder=5)
        ax.annotate(m["name"], (m["lon"], m["lat"]),
                    xytext=(7, 0), textcoords="offset points",
                    color="#ffb199", fontsize=8, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground="#0b0d12")])
    if s.get("district_mines"):
        ax.plot([], [], "s", mec="#ff8e6e", mfc="none", ms=8, mew=1.6,
                label="District deposit")

    type_style = {
        "city":       dict(marker="*", ms=15, color="#ffffff"),
        "town":       dict(marker="o", ms=7,  color="#cfe1ff"),
        "village":    dict(marker="o", ms=5,  color="#9fb3d1"),
        "hamlet":     dict(marker=".", ms=5,  color="#7d8ca3"),
        "suburb":     dict(marker="o", ms=5,  color="#9fb3d1"),
        "airport":    dict(marker="^", ms=8,  color="#7df0ff"),
        "aerodrome":  dict(marker="^", ms=8,  color="#7df0ff"),
        "place":      dict(marker=".", ms=5,  color="#7d8ca3"),
    }
    seen = set()
    for lm in landmarks:
        st = type_style.get(lm["type"], type_style["place"])
        ax.plot(lm["lon"], lm["lat"], st["marker"],
                color=st["color"], ms=st["ms"],
                mec="#0b0d12", mew=.8, zorder=6)
        if lm["type"] in ("city", "town", "airport", "aerodrome"):
            ax.annotate(lm["name"], (lm["lon"], lm["lat"]),
                        xytext=(8, 5), textcoords="offset points",
                        color="white", fontsize=9.5,
                        fontweight="bold" if lm["type"] == "city" else "normal",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5,
                                                    foreground="#0b0d12")])
        if lm["type"] not in seen:
            ax.plot([], [], st["marker"], color=st["color"], ms=st["ms"],
                    mec="#0b0d12", mew=.8, label=lm["type"].capitalize())
            seen.add(lm["type"])

    for p in picked:
        ax.add_patch(Circle((p["lon"], p["lat"]), 0.013, fill=False,
                            ec="#5af0a0", lw=2.4, zorder=8))
        ax.add_patch(Circle((p["lon"], p["lat"]), 0.024, fill=False,
                            ec="#5af0a0", lw=.9, alpha=.45, zorder=8))
        ax.annotate(f"#{p['rank']}", (p["lon"], p["lat"]),
                    xytext=(11, 7), textcoords="offset points",
                    color="#5af0a0", fontsize=11, fontweight="bold",
                    zorder=9,
                    path_effects=[pe.withStroke(linewidth=2.5,
                                                foreground="#0b0d12")])
    ax.plot([], [], "o", mec="#5af0a0", mfc="none", mew=2.2, ms=11,
            label=f"Drill target ({len(picked)})")

    cb = plt.colorbar(im, ax=ax, shrink=.7, pad=.02)
    cb.set_label("Prospectivity M(x,z) · azimuthal weight", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    cb.outline.set_edgecolor("#2a3144")

    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel("Longitude", color="#a8b3c8")
    ax.set_ylabel("Latitude",  color="#a8b3c8")
    ax.tick_params(colors="#a8b3c8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a3144")
    ax.set_title(f"Drill Targets — {s['name']} · {s['commodity']} · "
                 f"{RADIUS_KM:.0f} km radius",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    leg = ax.legend(loc="upper right", facecolor="#141826",
                    edgecolor="#2a3144", labelcolor="white",
                    framealpha=.95, fontsize=9)
    for t in leg.get_texts():
        t.set_color("white")

    lines = [f"{'#':>2} {'lat':>9} {'lon':>9} {'r_km':>5} "
             f"{'depth_m':>7} {'az':>4} {'score':>7}"]
    for p in picked:
        lines.append(f"{p['rank']:>2} {p['lat']:>9.4f} {p['lon']:>9.4f} "
                     f"{p['dist_km']:>5.1f} {p['depth_m']:>7.0f} "
                     f"{p['az_deg']:>4.0f} {p['score']:>7.4f}")
    ax.text(.012, .012, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, family="monospace", color="#5af0a0", va="bottom",
            bbox=dict(boxstyle="round,pad=0.55", fc="#10131bee",
                      ec="#2a3144", lw=1))
    ax.text(.5, -.075,
            "2D radial PINN. Azimuthal modulation is heuristic, not learned. "
            "Advisory only.",
            transform=ax.transAxes, ha="center",
            color="#7c8aa3", fontsize=8.5)

    plt.tight_layout()
    p_out = out_dir / "targets_map.png"
    plt.savefig(p_out, dpi=170, facecolor="#0b0d12", bbox_inches="tight")
    plt.close()
    (out_dir / "targets.json").write_text(json.dumps(picked, indent=2))
    (out_dir / "landmarks.json").write_text(
        json.dumps(landmarks, indent=2, ensure_ascii=False))
    print(f"saved {p_out}")
    print(f"\nTop {len(picked)} drill targets — {s['name']}:")
    for p in picked:
        print(f"  #{p['rank']}  ({p['lat']:.4f}, {p['lon']:.4f})  "
              f"r={p['dist_km']:>5.1f} km  depth={p['depth_m']:>5.0f} m  "
              f"score={p['score']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT, choices=list(SITES.keys()))
    main(ap.parse_args().site)