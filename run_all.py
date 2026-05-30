"""Driver: fetch -> train -> viz -> targets, for one or many sites."""
import argparse, subprocess, sys
from sites import SITES


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!!! FAILED: {' '.join(cmd)}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=list(SITES.keys()),
                    help="site keys to process")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=2000)
    a = ap.parse_args()

    for s in a.sites:
        if s not in SITES:
            print(f"unknown site '{s}', skipping"); continue
        print(f"\n========== {SITES[s]['name']} ==========")
        if not a.skip_fetch:
            run([sys.executable, "fetch.py", "--site", s])
        if not a.skip_train:
            run([sys.executable, "pinn.py", "--site", s,
                 "--epochs", str(a.epochs)])
        run([sys.executable, "viz.py",     "--site", s])
        run([sys.executable, "targets.py", "--site", s])


if __name__ == "__main__":
    main()