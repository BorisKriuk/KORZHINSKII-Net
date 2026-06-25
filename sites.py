"""sites.py — registry of Russian study sites for the PINN demo.

v5: SCOPE CUT.
  The model is infiltration-metasomatism / reactive-transport physics. That
  physics is only defensible for fluid-driven, wall-rock-replacement systems.
  The following deposits are therefore NOT in this registry anymore, because
  they are governed by other processes and including them was scientifically
  indefensible:
    - Norilsk / Pechenga : magmatic Ni-Cu-PGE sulfide (magma differentiation,
                            sulfide liquid immiscibility) — not infiltration MTS
    - Mirny              : kimberlite (mantle-sourced diatreme) — not MTS
  Retained sites are all replacement / fluid-rock systems:
    - Udokan     : sediment-hosted Cu  (red-bed / reduction-front replacement)
    - Sukhoi Log : orogenic Au         (fluid-rock alteration along structures)
    - Natalka    : orogenic Au         (same class as Sukhoi Log)

v3: heat_flow_mwm2 from published continental HF compilations.
Sources: Duchkov & Sokolova 2014 (Sib craton), Smirnov 2008 (NE Russia).
Continental reference HF_REF = 65 mW/m^2 used to normalize bottom-T target.
"""

HF_REF_MWM2 = 65.0  # continental reference for normalization

SITES = {
    "udokan": {
        "name": "Udokan",
        "lat": 56.9700, "lon": 118.0700, "radius_km": 70.0,
        "commodity": "Cu (sandstone-hosted)",
        "heat_flow_mwm2": 48.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Novaya Chara", "lat": 56.8050, "lon": 118.2700, "type": "town"},
            {"name": "Chara",        "lat": 56.9050, "lon": 118.2667, "type": "village"},
        ],
        "district_mines": [{"name": "Udokan deposit", "lat": 56.9700, "lon": 118.0700}],
        "seed_mines": [
            (56.9700, 118.0700), (57.0050, 118.2000), (56.8550, 117.9000),
            (57.0500, 118.0000), (56.9000, 118.3000), (56.7800, 118.1500),
        ],
        "fallback_column": [(0, 1500), (0, 1500), (4, 500), (1, 800), (4, 700)],
    },
    "sukhoi_log": {
        "name": "Sukhoi Log",
        "lat": 58.1800, "lon": 115.4700, "radius_km": 60.0,
        "commodity": "Au (orogenic)",
        "heat_flow_mwm2": 45.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Bodaybo",   "lat": 57.8500, "lon": 114.1900, "type": "town"},
            {"name": "Kropotkin", "lat": 58.5000, "lon": 115.1167, "type": "village"},
        ],
        "district_mines": [{"name": "Sukhoi Log", "lat": 58.1800, "lon": 115.4700}],
        "seed_mines": [
            (58.1800, 115.4700), (58.2600, 115.5500), (58.1000, 115.4000),
            (58.3000, 115.6000), (58.0800, 115.5500), (58.2200, 115.3500),
        ],
        "fallback_column": [(0, 1500), (0, 1500), (4, 500), (2, 500), (4, 1000)],
    },
    "natalka": {
        "name": "Natalka (Kolyma)",
        "lat": 61.8200, "lon": 147.4500, "radius_km": 90.0,
        "commodity": "Au (orogenic)",
        "heat_flow_mwm2": 60.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Omchak",  "lat": 61.6833, "lon": 147.7333, "type": "village"},
            {"name": "Susuman", "lat": 62.7833, "lon": 148.1500, "type": "town"},
        ],
        "district_mines": [
            {"name": "Natalka", "lat": 61.8200, "lon": 147.4500},
            {"name": "Pavlik",  "lat": 61.7700, "lon": 147.6500},
        ],
        "seed_mines": [
            (61.8200, 147.4500), (61.7700, 147.6500), (62.0700, 148.3000),
            (61.9000, 148.1000), (61.7000, 147.3000), (61.6500, 147.8000),
        ],
        "fallback_column": [(0, 1500), (0, 1500), (2, 500), (4, 500), (4, 1000)],
    },
    "olimpiada": {
        "name": "Olimpiada (Yenisei Ridge)",
        "lat": 60.0400, "lon": 92.6500, "radius_km": 70.0,
        "commodity": "Au (orogenic/disseminated)",
        "heat_flow_mwm2": 50.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Severo-Yeniseysky", "lat": 60.3800, "lon": 93.0300, "type": "town"},
        ],
        "district_mines": [{"name": "Olimpiada", "lat": 60.0400, "lon": 92.6500}],
        "seed_mines": [
            (60.0400, 92.6500), (60.2000, 93.0000), (60.0200, 92.7000),
            (60.1000, 92.8000), (60.1500, 92.5500), (59.9800, 92.6000),
        ],
        "fallback_column": [(0, 1500), (0, 1500), (4, 500), (2, 500), (4, 1000)],
    },
    "berezovskoye": {
        "name": "Berezovskoye (Urals)",
        "lat": 56.9100, "lon": 60.8100, "radius_km": 40.0,
        "commodity": "Au (orogenic vein)",
        "heat_flow_mwm2": 50.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Berezovsky",   "lat": 56.9100, "lon": 60.8100, "type": "town"},
            {"name": "Yekaterinburg","lat": 56.8389, "lon": 60.6057, "type": "city"},
        ],
        "district_mines": [{"name": "Berezovskoye", "lat": 56.9100, "lon": 60.8100}],
        "seed_mines": [
            (56.9100, 60.8100), (56.9300, 60.8300), (56.8900, 60.7800),
            (56.9500, 60.8000), (56.8700, 60.8500), (56.9000, 60.7600),
        ],
        "fallback_column": [(0, 1200), (0, 1000), (4, 400), (2, 400), (4, 1000)],
    },
    "vorontsovskoye": {
        "name": "Vorontsovskoye (Carlin-type, Urals)",
        "lat": 59.5200, "lon": 60.3200, "radius_km": 40.0,
        "commodity": "Au (carbonate-replacement)",
        "heat_flow_mwm2": 50.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Krasnoturyinsk", "lat": 59.7600, "lon": 60.1900, "type": "town"},
        ],
        "district_mines": [{"name": "Vorontsovskoye", "lat": 59.5200, "lon": 60.3200}],
        "seed_mines": [
            (59.5200, 60.3200), (59.5500, 60.3000), (59.5000, 60.3500),
            (59.4800, 60.2800), (59.5700, 60.3300), (59.5300, 60.3700),
        ],
        "fallback_column": [(0, 1200), (0, 1000), (4, 400), (2, 400), (4, 1000)],
    },
    "dalnegorsk": {
        "name": "Dalnegorsk (skarn district)",
        "lat": 44.5500, "lon": 135.5700, "radius_km": 50.0,
        "commodity": "B / Pb-Zn (Ca-skarn)",
        "heat_flow_mwm2": 65.0, "metasomatic_relevant": True,
        "landmarks": [
            {"name": "Dalnegorsk", "lat": 44.5560, "lon": 135.5670, "type": "town"},
        ],
        "district_mines": [{"name": "Dalnegorsk B-skarn", "lat": 44.5500, "lon": 135.5700}],
        "seed_mines": [
            (44.5500, 135.5700), (44.5000, 135.5000), (44.6000, 135.6000),
            (44.6500, 135.5500), (44.4500, 135.6200), (44.5800, 135.4800),
        ],
        "fallback_column": [(0, 1000), (0, 1000), (2, 600), (4, 400), (4, 1000)],
    },
}

DEFAULT = "udokan"

# All retained sites are in scope by construction; kept for API stability.
METASOMATIC_SITES = [k for k, v in SITES.items()
                     if v.get("metasomatic_relevant")]
OUT_OF_SCOPE_SITES = [k for k, v in SITES.items()
                      if not v.get("metasomatic_relevant")]  # now empty