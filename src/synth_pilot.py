"""Write a synthetic pilot capture (calibration targets + fabric board, see gwps/calib_synth.py)."""
import argparse

from gwps.calib_synth import make_pilot

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, choices=[3, 6], default=6)
    ap.add_argument("--swap-sigma-mm", type=float, default=0.02, help="mirror/matte ball seat repeatability (1 sigma per axis)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distorted", action="store_true", help="render every capture through the distorted lens")
    a = ap.parse_args()
    print(make_pilot(a.out, a.positions, a.swap_sigma_mm / 1000, not a.no_noise, a.seed, distorted=a.distorted))
