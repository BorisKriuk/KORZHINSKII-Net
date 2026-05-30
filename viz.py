"""viz.py — render the trained prospectivity field per site (v2).

Updated for pinn.py v2:
  - lith_at(x, z, column)   (was lith_at(z, column))
  - mines is List[float]    (was List[Tuple[float, float]])
  - uses compute_M() helper
  - mines plotted at the depth of their per-column M maximum
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pinn import (PINN, compute_M, lith_at,
                  AR, N_LITH, LITH_NAMES, DEVICE)
from sites import SITES, DEFAULT


def main(site_key):
    s = SITES[site_key]
    out = Path("output") / site_key
    ckpt = torch.load(out / "model.pt", map_location=DEVICE, weights_only=False)
    model = PINN().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    column, mines = ckpt["column"], ckpt["mines"]

    nx, nz = 240, 120
    xs = torch.linspace(0, AR, nx, device=DEVICE)
    zs = torch.linspace(0, 1,  nz, device=DEVICE)
    X, Z = torch.meshgrid(xs, zs, indexing="xy")
    x, z = X.flatten(), Z.flatten()

    # M field
    Mflat = compute_M(model, x, z, column)
    Mg = Mflat.cpu().numpy().reshape(nz, nx)

    # T field
    with torch.no_grad():
        T, _, _ = model.fields(x.detach(), z.detach())
    Tg = T.cpu().numpy().reshape(nz, nx)

    # lithology field (now x,z aware -> dipping layers visible)
    Lflat = lith_at(x, z, column)
    Lg = Lflat.cpu().numpy().reshape(nz, nx)

    # ----- per-mine optimal depth (where M peaks along its column) -----
    mine_xz = []
    if len(mines) > 0:
        n_zc = 80
        zc = torch.linspace(0.02, 0.98, n_zc, device=DEVICE)
        for mx_ in mines:
            xc = torch.full_like(zc, float(mx_))
            Mc = compute_M(model, xc, zc, column).cpu().numpy()
            iz = int(np.argmax(Mc))
            mine_xz.append((float(mx_), float(zc[iz].item())))

    # ----- plot -----
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    cmap_l = ListedColormap(["#d2b48c", "#3a3a3a", "#c0392b",
                             "#f1c40f", "#bbbbbb"])
    ax[0].imshow(Lg, extent=[0, AR, 1, 0], aspect="auto",
                 cmap=cmap_l, vmin=0, vmax=N_LITH - 1)
    ax[0].set_title(f"Lithology — {s['name']} ({s['commodity']})")
    ax[0].set_ylabel("z")
    # tiny legend for lithology classes
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cmap_l.colors[i], label=LITH_NAMES[i])
               for i in range(N_LITH)]
    ax[0].legend(handles=handles, loc="upper right",
                 fontsize=7, framealpha=0.85)

    im1 = ax[1].imshow(Tg, extent=[0, AR, 1, 0], aspect="auto", cmap="inferno")
    ax[1].set_title("Temperature T")
    ax[1].set_ylabel("z")
    plt.colorbar(im1, ax=ax[1], shrink=0.8)

    Ml = np.log10(Mg + 1e-8)
    im2 = ax[2].imshow(Ml, extent=[0, AR, 1, 0], aspect="auto", cmap="viridis")
    ax[2].set_title("Mineralization potential  log10 M(x,z)")
    ax[2].set_xlabel("x"); ax[2].set_ylabel("z")
    plt.colorbar(im2, ax=ax[2], shrink=0.8)
    for mx_, mz_ in mine_xz:
        ax[2].plot(mx_, mz_, "rx", ms=10, mew=2)

    plt.tight_layout()
    p = out / "prospectivity.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"saved {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT, choices=list(SITES.keys()))
    main(ap.parse_args().site)