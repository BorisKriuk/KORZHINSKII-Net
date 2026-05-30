"""sites.py — registry of Russian study sites for the PINN demo.

v3: added heat_flow_mwm2 from published continental HF compilations.
Sources: Duchkov & Sokolova 2014 (Sib craton), Slagstad 2008 (Baltic),
Khutorskoy 2013 (Norilsk), Smirnov 2008 (NE Russia). Continental
reference HF_REF = 65 mW/m^2 used to normalize bottom-T target.
"""

HF_REF_MWM2 = 65.0  # continental reference for normalization

SITES = {
    "norilsk": {
        "name": "Norilsk",
        "lat": 69.3535, "lon": 88.2027, "radius_km": 60.0,
        "commodity": "Ni-Cu-PGE",
        "heat_flow_mwm2": 55.0,
        "landmarks": [
            {"name": "Norilsk",        "lat": 69.3535, "lon": 88.2027, "type": "city"},
            {"name": "Talnakh",        "lat": 69.4865, "lon": 88.3972, "type": "town"},
            {"name": "Kayerkan",       "lat": 69.3833, "lon": 87.7500, "type": "town"},
            {"name": "Snezhnogorsk",   "lat": 69.1903, "lon": 87.7497, "type": "town"},
            {"name": "Alykel Airport", "lat": 69.3110, "lon": 87.3322, "type": "airport"},
        ],
        "district_mines": [
            {"name": "Oktyabrsky",  "lat": 69.5070, "lon": 88.4170},
            {"name": "Taimyrsky",   "lat": 69.5000, "lon": 88.4060},
            {"name": "Komsomolsky", "lat": 69.4920, "lon": 88.4010},
            {"name": "Skalisty",    "lat": 69.5020, "lon": 88.4280},
            {"name": "Mayak",       "lat": 69.4830, "lon": 88.3640},
            {"name": "Norilsk-1",   "lat": 69.3640, "lon": 88.2050},
        ],
        "fallback_column": [(0, 500), (1, 2000), (2, 1000), (3, 1000), (4, 500)],
    },
    "pechenga": {
        "name": "Pechenga (Kola)",
        "lat": 69.4117, "lon": 30.2167, "radius_km": 50.0,
        "commodity": "Ni-Cu sulphide",
        "heat_flow_mwm2": 42.0,
        "landmarks": [
            {"name": "Nikel",      "lat": 69.4117, "lon": 30.2167, "type": "town"},
            {"name": "Zapolyarny", "lat": 69.4167, "lon": 30.8000, "type": "town"},
            {"name": "Pechenga",   "lat": 69.5500, "lon": 31.2333, "type": "village"},
        ],
        "district_mines": [
            {"name": "Zhdanovskoye", "lat": 69.4000, "lon": 30.7500},
            {"name": "Zapolyarnoye", "lat": 69.4350, "lon": 30.7700},
        ],
        "fallback_column": [(0, 400), (1, 1500), (2, 1500), (4, 500), (1, 1100)],
    },
    "udokan": {
        "name": "Udokan",
        "lat": 56.9700, "lon": 118.0700, "radius_km": 60.0,
        "commodity": "Cu (sandstone-hosted)",
        "heat_flow_mwm2": 48.0,
        "landmarks": [
            {"name": "Novaya Chara", "lat": 56.8050, "lon": 118.2700, "type": "town"},
            {"name": "Chara",        "lat": 56.9050, "lon": 118.2667, "type": "village"},
        ],
        "district_mines": [
            {"name": "Udokan deposit", "lat": 56.9700, "lon": 118.0700},
        ],
        "fallback_column": [(0, 1500), (0, 1500), (4, 500), (1, 800), (4, 700)],
    },
    "sukhoi_log": {
        "name": "Sukhoi Log",
        "lat": 58.1000, "lon": 115.5000, "radius_km": 50.0,
        "commodity": "Au (orogenic)",
        "heat_flow_mwm2": 45.0,
        "landmarks": [
            {"name": "Bodaybo",   "lat": 57.8500, "lon": 114.1900, "type": "town"},
            {"name": "Kropotkin", "lat": 58.5000, "lon": 115.1167, "type": "village"},
        ],
        "district_mines": [
            {"name": "Sukhoi Log", "lat": 58.1000, "lon": 115.5000},
        ],
        "fallback_column": [(0, 1500), (0, 1500), (4, 500), (2, 500), (4, 1000)],
    },
    "natalka": {
        "name": "Natalka (Kolyma)",
        "lat": 61.8200, "lon": 147.4500, "radius_km": 50.0,
        "commodity": "Au",
        "heat_flow_mwm2": 60.0,
        "landmarks": [
            {"name": "Omchak",  "lat": 61.6833, "lon": 147.7333, "type": "village"},
            {"name": "Susuman", "lat": 62.7833, "lon": 148.1500, "type": "town"},
        ],
        "district_mines": [
            {"name": "Natalka", "lat": 61.8200, "lon": 147.4500},
            {"name": "Pavlik",  "lat": 61.7700, "lon": 147.6500},
        ],
        "fallback_column": [(0, 1500), (0, 1500), (2, 500), (4, 500), (4, 1000)],
    },
    "mirny": {
        "name": "Mirny (Yakutia)",
        "lat": 62.5350, "lon": 113.9650, "radius_km": 60.0,
        "commodity": "Diamond (kimberlite)",
        "heat_flow_mwm2": 40.0,
        "landmarks": [
            {"name": "Mirny",  "lat": 62.5350, "lon": 113.9650, "type": "city"},
            {"name": "Aikhal", "lat": 65.9333, "lon": 111.5000, "type": "town"},
        ],
        "district_mines": [
            {"name": "Mir pipe",            "lat": 62.5300, "lon": 113.9930},
            {"name": "Internatsionalnaya",  "lat": 62.4670, "lon": 113.9750},
        ],
        "fallback_column": [(0, 500), (0, 1500), (3, 500), (2, 1500), (4, 1000)],
    },
}

DEFAULT = "norilsk"