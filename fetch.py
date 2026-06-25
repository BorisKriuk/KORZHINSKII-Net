"""Per-site downloader (v3): Macrostrat, OSM mines, OSM towns,
NASA POWER, USGS earthquakes, Heat-flow lookup.

New free, key-less endpoints:
  - USGS FDSN earthquake catalog (https://earthquake.usgs.gov/fdsnws)
  - Embedded heat-flow compilation per site (literature values).
    Can be overridden by dropping a CSV at data/<site>/heat_flow.csv
    with column 'hf_mwm2' (we average it).
"""
import argparse, json, csv
from pathlib import Path
import requests
from sites import SITES, DEFAULT

HDR = {"User-Agent": "pinn-mining/0.5"}

# Heat-flow compilation (mW/m^2) — published continental HF for each site.
HEAT_FLOW_DB = {
    "norilsk":        {"hf_mwm2": 55.0, "src": "Khutorskoy 2013"},
    "pechenga":       {"hf_mwm2": 42.0, "src": "Slagstad 2008"},
    "udokan":         {"hf_mwm2": 48.0, "src": "Duchkov 2014"},
    "sukhoi_log":     {"hf_mwm2": 45.0, "src": "Duchkov 2010"},
    "natalka":        {"hf_mwm2": 60.0, "src": "Smirnov 2008"},
    "mirny":          {"hf_mwm2": 40.0, "src": "Duchkov 2014"},
    "olimpiada":      {"hf_mwm2": 50.0, "src": "Duchkov 2014"},
    "berezovskoye":   {"hf_mwm2": 50.0, "src": "Urals compilation"},
    "vorontsovskoye": {"hf_mwm2": 50.0, "src": "Urals compilation"},
    "dalnegorsk":     {"hf_mwm2": 65.0, "src": "Far East compilation"},
}


def fetch_macrostrat_units(lat, lon):
    url = (f"https://macrostrat.org/api/v2/units?lat={lat}&lng={lon}"
           f"&response=long&format=json")
    return requests.get(url, timeout=60, headers=HDR).json()


def fetch_overpass_mines(lat, lon, radius_m):
    q = f"""
    [out:json][timeout:90];
    ( node["man_made"="mineshaft"](around:{radius_m},{lat},{lon});
      node["man_made"="adit"]      (around:{radius_m},{lat},{lon});
      node["industrial"="mine"]    (around:{radius_m},{lat},{lon});
      way ["industrial"="mine"]    (around:{radius_m},{lat},{lon});
      way ["landuse"="quarry"]     (around:{radius_m},{lat},{lon});
      way ["landuse"="industrial"]["name"~"mine|mining|rudnik|шахт|рудник|обогат",i]
            (around:{radius_m},{lat},{lon});
    ); out center tags;"""
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": q}, timeout=180, headers=HDR)
    r.raise_for_status()
    return r.json()


def fetch_overpass_towns(lat, lon, radius_m):
    q = f"""[out:json][timeout:60];
    ( node["place"~"city|town|village|hamlet|suburb"](around:{radius_m},{lat},{lon});
      node["aeroway"="aerodrome"](around:{radius_m},{lat},{lon});
    ); out tags center;"""
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": q}, timeout=90, headers=HDR)
    r.raise_for_status()
    out = []
    for e in r.json().get("elements", []):
        t = e.get("tags", {})
        nm = t.get("name:en") or t.get("name")
        if not nm:
            continue
        kind = t.get("place") or t.get("aeroway") or "place"
        out.append({"name": nm, "lat": float(e["lat"]),
                    "lon": float(e["lon"]), "type": kind})
    return out


def fetch_power(lat, lon):
    url = ("https://power.larc.nasa.gov/api/temporal/climatology/point"
           f"?parameters=T2M&community=AG&longitude={lon}&latitude={lat}"
           "&format=JSON")
    return requests.get(url, timeout=60, headers=HDR).json()


def fetch_earthquakes(lat, lon, radius_km,
                      min_mag=2.5, start="1980-01-01"):
    """USGS FDSN — free, no key. Returns GeoJSON FeatureCollection."""
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query"
           f"?format=geojson&latitude={lat}&longitude={lon}"
           f"&maxradiuskm={int(radius_km)}&starttime={start}"
           f"&minmagnitude={min_mag}&limit=20000")
    r = requests.get(url, timeout=120, headers=HDR)
    r.raise_for_status()
    return r.json()


def fetch_heat_flow(site_key, fallback_mwm2, data_dir):
    """Embedded literature value, optionally overridden by
    data/<site>/heat_flow.csv (column 'hf_mwm2')."""
    base = HEAT_FLOW_DB.get(site_key,
                            {"hf_mwm2": fallback_mwm2, "src": "site fallback"})
    csv_path = data_dir / "heat_flow.csv"
    if csv_path.exists():
        try:
            vals = []
            with csv_path.open() as fh:
                rd = csv.DictReader(fh)
                for row in rd:
                    try:
                        vals.append(float(row["hf_mwm2"]))
                    except Exception:
                        pass
            if vals:
                base = {"hf_mwm2": sum(vals) / len(vals),
                        "src": f"user CSV (n={len(vals)})"}
        except Exception:
            pass
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT, choices=list(SITES.keys()))
    args = ap.parse_args()
    s = SITES[args.site]
    out = Path("data") / args.site
    out.mkdir(parents=True, exist_ok=True)
    lat, lon = s["lat"], s["lon"]
    R_m  = s["radius_km"] * 1000
    R_km = s["radius_km"]

    print(f"=== Fetching: {s['name']} ({lat}, {lon}) "
          f"R={R_km:.0f} km ===")

    jobs = [
        ("units",        lambda: fetch_macrostrat_units(lat, lon),
         "Macrostrat units"),
        ("mines",        lambda: fetch_overpass_mines(lat, lon, R_m),
         "OSM mines"),
        ("towns",        lambda: fetch_overpass_towns(lat, lon, R_m),
         "OSM towns"),
        ("power",        lambda: fetch_power(lat, lon),
         "NASA POWER"),
        ("earthquakes",  lambda: fetch_earthquakes(lat, lon, R_km),
         "USGS earthquakes"),
        ("heat_flow",    lambda: fetch_heat_flow(args.site,
                                                 s.get("heat_flow_mwm2", 55.0),
                                                 out),
         "Heat-flow lookup"),
    ]
    for tag, fn, name in jobs:
        try:
            print(f"  - {name} ...")
            data = fn()
            (out / f"{tag}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2))
            if tag == "earthquakes":
                n = len(data.get("features", []))
                print(f"      {n} events")
            if tag == "heat_flow":
                print(f"      hf={data['hf_mwm2']:.1f} mW/m^2  ({data['src']})")
        except Exception as e:
            print(f"    FAIL: {e}")
    print(f"OK -> {out}/")


if __name__ == "__main__":
    main()