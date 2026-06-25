# KORZHINSKII-Net (v2)
A Physics-Informed Neural Network for Mineral Prospectivity Modelling

> Named after Dmitri Sergeyevich Korzhinskii (1899–1985) — founder of physico-chemical
> petrology and the theory of infiltration metasomatism, the physical scaffold this
> network is constrained by.

## Overview
KORZHINSKII-Net couples a 2D radial physics-informed neural network (PINN) with
lithology-aware reactive transport equations to predict subsurface mineralization
potential M(x, z). The network jointly solves for three coupled fields:

| Field | Quantity      | Governing physics                     |
|-------|---------------|---------------------------------------|
| T     | Temperature   | Advection–diffusion heat transport    |
| P     | Pressure      | Darcy flow                            |
| C     | Concentration | Reaction-rate-limited solubility      |

## Ore Provinces Covered
| Site       | Commodity          | Tectonic setting     |
|------------|--------------------|----------------------|
| Norilsk    | Ni–Cu–PGE          | Siberian Traps       |
| Pechenga   | Ni–Cu sulphide     | Baltic Shield (Kola) |
| Udokan     | Cu (sandstone)     | Aldan–Stanovoy       |
| Sukhoi Log | Au (orogenic)      | Baikal–Patom belt    |
| Natalka    | Au (orogenic)      | Yana–Kolyma          |
| Mirny      | Diamond (kimberlite)| Siberian craton     |

Evaluated under leave-one-deposit-out (LODO) and K-fold cross-validation across
4 commodity classes.

## Pipeline
fetch.py -> pinn.py -> viz.py -> targets.py
pinn.py  -> evaluate_lodo.py
all      -> run_all.py (end-to-end driver)

| Module           | Role                                           |
|------------------|------------------------------------------------|
| fetch.py         | Pulls Macrostrat, OSM, USGS, NASA POWER        |
| pinn.py          | Trains PINN per site with proxy modulators     |
| viz.py           | Renders T, M, lithology cross-sections         |
| targets.py       | Generates geographic drill-target maps         |
| evaluate_lodo.py | Benchmarks vs 7 ML baselines                   |
| run_all.py       | End-to-end driver                              |

## Quick Start
# Install
python -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib scikit-learn requests

# Run end-to-end
python run_all.py --epochs 2000

# Run benchmark (LODO + 5-fold)
python evaluate_lodo.py \
    --sites all \
    --epochs 800 \
    --neg-mode hard \
    --r-inner 0.4 \
    --r-outer 2.0 \
    --kfold 5 \
    --tag fair_5fold_all

## Method
The model approximates three coupled fields with a multi-head MLP:
    (T, P, C) = f_theta(x, z)

constrained by:
  Darcy flow:                      q = -(k/mu) * grad(P)
  Advection–diffusion heat:        rho*c_p * q . grad(T) = div(lambda * grad(T))
  Softplus-saturated reaction:     R(T, C) = softplus( alpha * k(T) * [C - C_eq(T, l)] )

depending on lithology-specific solubility and proxy modulators:
  - Faults
  - Seismicity
  - Lithological contacts
  - Deep intrusive roots

Mineralization field (prediction target):
    M(x, z) = R( T(x,z), C(x,z) )

## Baselines
| Model | Description                          |
|-------|--------------------------------------|
| PROXY | M = k_mod(x) * s_z(z) — no training  |
| LR    | Logistic regression                  |
| RF    | Random forest                        |
| ET    | Extra trees                          |
| GB    | Gradient boosting                    |
| KNN   | k-nearest neighbors                  |
| SVM   | Support vector (RBF kernel)          |
| MLP   | Multi-layer perceptron               |

All baselines use the same CV folds as the PINN, with hard ring negatives and
jittered positive z to prevent leakage.

## Project Structure
KORZHINSKII-Net/
├── fetch.py              # Data acquisition
├── pinn.py               # PINN training
├── viz.py                # Field rendering
├── targets.py            # Drill-target generation
├── evaluate_lodo.py      # LODO + K-fold benchmark
├── run_all.py            # End-to-end pipeline
├── data/                 # Cached site data
├── outputs/              # Figures + target maps
└── LICENSE

## Acknowledgements
This work was made possible through the scientific guidance and petrological
expertise of:
  - Alexander Simakin — Institute of Experimental Mineralogy, RAS
  - Safonov Oleg      — Institute of Experimental Mineralogy, RAS

## Citation
@software{korzhinskii_net_2026,
  author = {Kriuk, Boris},
  title  = {KORZHINSKII-Net: a physics-informed neural network for mineral prospectivity modelling},
  year   = {2026},
  url    = {https://github.com/BorisKriuk/KORZHINSKII-Net}
}

## License
Released under the Apache License 2.0 — see LICENSE.
