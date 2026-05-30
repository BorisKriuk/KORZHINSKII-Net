# KORZHINSKII-Net

**A Physics-Informed Neural Network for Mineral Prospectivity Modelling
in Russian Ore Provinces.**

Named after Dmitri Sergeyevich Korzhinskii (1899–1985), founder of
physico-chemical petrology and the theory of infiltration metasomatism —
the physical scaffold this network is constrained by.

---

## Overview

KORZHINSKII-Net couples a 2D radial physics-informed neural network
(PINN) with lithology-aware reactive transport equations to predict
subsurface mineralization potential \(M(x, z)\). The network jointly
solves for temperature \(T\), pressure \(P\), and metal concentration
\(C\) under PDE constraints derived from Darcy flow, advection-diffusion
heat transport, and reaction-rate-limited solubility.

The framework is evaluated leave-one-deposit-out (LODO) and via K-fold
cross-validation across six Russian ore districts spanning four
commodity classes:

| Site         | Commodity            | Tectonic setting         |
|--------------|----------------------|--------------------------|
| Norilsk      | Ni-Cu-PGE            | Siberian Traps           |
| Pechenga     | Ni-Cu sulphide       | Baltic Shield (Kola)     |
| Udokan       | Cu (sandstone-hosted)| Aldan-Stanovoy           |
| Sukhoi Log   | Au (orogenic)        | Baikal-Patom belt        |
| Natalka      | Au (orogenic)        | Yana-Kolyma              |
| Mirny        | Diamond (kimberlite) | Siberian craton          |

---

## Pipeline

```
fetch.py    →  data acquisition  (Macrostrat, OSM, USGS, NASA POWER)
pinn.py     →  PINN training      (per site, with proxy modulators)
viz.py      →  field rendering    (T, M, lithology cross-sections)
targets.py  →  drill-target maps  (geographic, top-K candidates)
evaluate_lodo.py → LODO / K-fold benchmark vs 7 ML baselines
run_all.py  →  end-to-end driver
```

---

## Quick start

```bash
# install
python -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib scikit-learn requests

# fetch data and train all sites
python run_all.py --epochs 2000

# benchmark vs baselines (5-fold CV, hard negatives)
python evaluate_lodo.py --sites all --epochs 800 \
    --neg-mode hard --r-inner 0.4 --r-outer 2.0 \
    --kfold 5 --tag fair_5fold_all
```

---

## Method (brief)

The model approximates three coupled fields with a multi-head MLP:

```
(T, P, C) = f_θ(x, z)
```

constrained by the residuals

- Darcy: \(\nabla \cdot \mathbf{q} = 0\), \(\mathbf{q} = -k(\nabla P - \mathrm{Ra}\, T \hat{z})\)
- Heat: \(\mathbf{q} \cdot \nabla T - \mathrm{Pe}_T^{-1} \nabla^2 T = Q\)
- Tracer: \(\mathbf{q} \cdot \nabla C - \mathrm{Pe}_C^{-1} \nabla^2 C = -R(T, C)\)

with reaction rate \(R\) softplus-saturated by lithology-dependent
solubility and modulated by deposit-type-specific proxies (faults,
seismicity, lithological contacts, deep intrusive roots).

The mineralization field
\(M(x, z) = R(T(x,z), C(x,z))\)
is the prediction target.

---

## Baselines

- **PROXY**: \(M = k_{\mathrm{mod}}(x) \cdot s_z(z)\) — no training. Sanity floor.
- LR, RF, ET, GB, KNN, SVM (RBF), MLP — fed lithology one-hot, depth,
  contact-distance features. Same CV folds as PINN.

The benchmark explicitly removes geometric and vertical-coordinate leaks
(hard ring negatives, jittered positive z) so baseline gains reflect
genuine signal.

---

## Citation

If you use this code, please cite:

```bibtex
@software{korzhinskii_net_2026,
  author = {Kriuk, Boris},
  title  = {KORZHINSKII-Net: a physics-informed neural network for
            mineral prospectivity modelling},
  year   = {2026},
  url    = {https://github.com/BorisKriuk/KORZHINSKII-Net}
}
```

---

## License

Apache-2.0. See `LICENSE`.
