<!-- ====================== ANIMATED HEADER ====================== -->
<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d4a8c,50:00b4d8,100:7209b7&height=200&section=header&text=KORZHINSKII-Net&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Physics-Informed%20Neural%20Network%20for%20Mineral%20Prospectivity&descAlignY=58&descSize=16" />

<a href="https://github.com/BorisKriuk/KORZHINSKII-Net">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=800&color=00B4D8&center=true&vCenter=true&width=720&lines=Temperature+%2B+Pressure+%2B+Concentration+%E2%86%92+Mineralization;Reactive+transport+constrained+deep+learning;LODO+%2B+K-Fold+across+6+ore+provinces;Norilsk+%E2%80%A2+Pechenga+%E2%80%A2+Udokan+%E2%80%A2+Sukhoi+Log+%E2%80%A2+Natalka+%E2%80%A2+Mirny" alt="typing" />
</a>

<br/>

![version](https://img.shields.io/badge/version-2.0-7209b7?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-PINN-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-00b4d8?style=for-the-badge)

![stars](https://img.shields.io/github/stars/BorisKriuk/KORZHINSKII-Net?style=social)
![forks](https://img.shields.io/github/forks/BorisKriuk/KORZHINSKII-Net?style=social)

</div>

> Named after **Dmitri Sergeyevich Korzhinskii** (1899–1985) — founder of physico-chemical petrology and the theory of infiltration metasomatism, the physical scaffold this network is constrained by.

---

## Overview

**KORZHINSKII-Net** couples a 2D radial physics-informed neural network with lithology-aware reactive transport equations to predict subsurface mineralization potential **M(x, z)**. It jointly solves three coupled fields:

<div align="center">

| Field | Quantity | Governing Physics |
|:-----:|:---------|:------------------|
| **T** | Temperature | Advection–diffusion heat transport |
| **P** | Pressure | Darcy flow |
| **C** | Concentration | Reaction-rate-limited solubility |

</div>

---

## Pipeline

\`\`\`mermaid
flowchart LR
    A[fetch.py] -->|Macrostrat / OSM / USGS / NASA POWER| B[pinn.py]
    B -->|trained fields| C[viz.py]
    B --> D[evaluate_lodo.py]
    C --> E[targets.py]
    D --> F[run_all.py]
    E --> F
    style A fill:#ff6b1a,stroke:#fff,color:#fff
    style B fill:#00b4d8,stroke:#fff,color:#fff
    style C fill:#7209b7,stroke:#fff,color:#fff
    style D fill:#39ff7c,stroke:#000,color:#000
    style E fill:#ffd700,stroke:#000,color:#000
    style F fill:#ff6b9d,stroke:#fff,color:#fff
\`\`\`

---

## Ore Provinces Covered

<div align="center">

| Site | Commodity | Tectonic Setting |
|:-----|:----------|:-----------------|
| 🟠 **Norilsk** | Ni–Cu–PGE | Siberian Traps |
| 🔵 **Pechenga** | Ni–Cu sulphide | Baltic Shield (Kola) |
| 🟠 **Udokan** | Cu (sandstone) | Aldan–Stanovoy |
| 🟡 **Sukhoi Log** | Au (orogenic) | Baikal–Patom belt |
| 🟡 **Natalka** | Au (orogenic) | Yana–Kolyma |
| 🟣 **Mirny** | Diamond (kimberlite) | Siberian craton |

</div>

Evaluated under **leave-one-deposit-out (LODO)** and **K-fold cross-validation** across 4 commodity classes.

---

## Method

The model approximates three coupled fields with a multi-head MLP:

$$(T, P, C) = f_\theta(x, z)$$

constrained by the governing physics:

$$q = -\frac{k}{\mu}\,\nabla P \qquad\text{(Darcy flow)}$$

$$\rho\,c_p\,(q \cdot \nabla T) = \nabla \cdot (\lambda \nabla T) \qquad\text{(advection–diffusion)}$$

$$R(T, C) = \mathrm{softplus}\!\big(\alpha\,k(T)\,[\,C - C_{eq}(T, \ell)\,]\big)$$

The mineralization field — the prediction target — is:

$$\boxed{\,M(x, z) = R\big(T(x,z),\, C(x,z)\big)\,}$$

Driven by lithology-specific solubility and proxy modulators: faults, seismicity, lithological contacts, and deep intrusive roots.

---

## Baselines

<div align="center">

| Model | Description |
|:-----:|:------------|
| \`PROXY\` | M = k_mod(x)·s_z(z) — no training |
| \`LR\` | Logistic regression |
| \`RF\` | Random forest |
| \`ET\` | Extra trees |
| \`GB\` | Gradient boosting |
| \`KNN\` | k-nearest neighbors |
| \`SVM\` | Support vector (RBF kernel) |
| \`MLP\` | Multi-layer perceptron |

</div>

All baselines use the same CV folds as the PINN, with hard ring negatives and jittered positive z to prevent leakage.

---

## Quick Start

<details open>
<summary><b>Installation</b></summary>

\`\`\`bash
python -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib scikit-learn requests
\`\`\`
</details>

<details>
<summary><b>Run end-to-end</b></summary>

\`\`\`bash
python run_all.py --epochs 2000
\`\`\`
</details>

<details>
<summary><b>Run benchmark (LODO + 5-fold)</b></summary>

\`\`\`bash
python evaluate_lodo.py \
    --sites all \
    --epochs 800 \
    --neg-mode hard \
    --r-inner 0.4 \
    --r-outer 2.0 \
    --kfold 5 \
    --tag fair_5fold_all
\`\`\`
</details>

---

## Project Structure

\`\`\`
KORZHINSKII-Net/
├── fetch.py              # Data acquisition
├── pinn.py               # PINN training
├── viz.py                # Field rendering
├── targets.py            # Drill-target generation
├── evaluate_lodo.py      # LODO + K-fold benchmark
├── run_all.py            # End-to-end driver
├── data/                 # Cached site data
├── outputs/              # Figures + target maps
└── LICENSE
\`\`\`

---

## Acknowledgements

This work was made possible through the scientific guidance and petrological expertise of:

<div align="center">

| | |
|:--|:--|
| **Alexander Simakin** | Institute of Experimental Mineralogy, RAS |
| **Safonov Oleg** | Institute of Experimental Mineralogy, RAS |

</div>

---

## Citation

\`\`\`bibtex
@software{korzhinskii_net_2026,
  author = {Kriuk, Boris},
  title  = {KORZHINSKII-Net: a physics-informed neural network
            for mineral prospectivity modelling},
  year   = {2026},
  url    = {https://github.com/BorisKriuk/KORZHINSKII-Net}
}
\`\`\`

---

<div align="center">

Released under the **Apache License 2.0**

⭐ *If this helped your research, consider starring the repo*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:7209b7,50:00b4d8,100:0d4a8c&height=120&section=footer" />

</div>
