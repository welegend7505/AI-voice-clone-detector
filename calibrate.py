"""Recompute the anti-spoof calibration from the sample folders.

Scores every wav in samples/real/ (genuine human speech) and samples/fake/
(synthetic/cloned speech) with the AASIST model, then prints:
  - per-sample raw bonafide logits (so you can eyeball separation)
  - suggested CALIBRATION constants for models/anti_spoof.py

Usage:
    python calibrate.py            # uses samples/real and samples/fake
    python calibrate.py <real_dir> <fake_dir>

If the printed numbers differ from what's in models/anti_spoof.py, edit
CALIBRATION there to match. See samples/PROVENANCE.md for what the current
samples are and why they are only a preliminary calibration.
"""

import glob
import sys

from models import anti_spoof


def collect(dirpath):
    files = sorted(glob.glob(dirpath.rstrip("/\\") + "/*.wav"))
    if not files:
        sys.exit(f"No .wav files found in {dirpath} -- add samples first.")
    return files


def main():
    real_dir = sys.argv[1] if len(sys.argv) > 2 else "samples/real"
    fake_dir = sys.argv[2] if len(sys.argv) > 2 else "samples/fake"

    scores = {}
    for label, d in [("real", real_dir), ("fake", fake_dir)]:
        print(f"--- scoring {label} samples from {d} ---")
        vals = []
        for f in collect(d):
            s = anti_spoof.score_audio(f)["bonafide_score"]
            print(f"  {f}: {s:+.3f}")
            vals.append(s)
        scores[label] = vals

    avg_real = sum(scores["real"]) / len(scores["real"])
    avg_fake = sum(scores["fake"]) / len(scores["fake"])
    threshold = anti_spoof.calibrate_threshold(scores["real"], scores["fake"])

    print()
    print(f"avg real logit : {avg_real:+.3f}")
    print(f"avg fake logit : {avg_fake:+.3f}")
    print(f"separation     : {avg_real - avg_fake:.3f}")
    print()
    print("Suggested constants for models/anti_spoof.py CALIBRATION:")
    print(f'    "threshold": {threshold:.2f},')
    print(f'    "ramp_halfwidth": {max(abs(avg_real - threshold), abs(avg_fake - threshold)):.2f},')
    print("(ramp_halfwidth = distance from threshold to the nearer average,")
    print(" so both class averages land near risk 0.0 / 1.0.)")
    print()
    print("Sanity check on current samples:")
    for f, s in [(f, anti_spoof.score_audio(f)["bonafide_score"])
                 for f in collect(real_dir) + collect(fake_dir)]:
        r = anti_spoof.spoof_risk_from_logit(s)
        print(f"  {f}: logit {s:+.3f} -> spoof_risk {r:.3f}")


if __name__ == "__main__":
    main()
