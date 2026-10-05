"""Write a synthetic dataset (see gwps/synth.py)."""
import argparse

from gwps.synth import make_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["torso", "board"], default="torso")
    ap.add_argument("--variant", choices=["bump", "ripple", "bias", "vbias"], default=None)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_dataset(a.out, a.kind, a.variant, a.noise, a.seed))
