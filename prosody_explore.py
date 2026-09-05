"""Re-measure whether any prosody/voice-quality statistic separates real
from neural TTS on the stress set. Uses models/prosody.extract_features
(the exact function the live pipeline calls) -- see that module's
docstring for the 2026-09-06 verdict this produced.

Usage:  python prosody_explore.py
"""

import glob
import os

import numpy as np

from models.prosody import extract_features

KEYS = ["mean_f0", "f0_cv", "voiced_frac", "jitter_local",
        "syll_nuclei_rate", "pause_frac"]


def main():
    wavs = sorted(w for w in glob.glob(os.path.join("samples", "stress", "*.wav"))
                  if not os.path.basename(w).startswith("real_enroll"))
    groups = {"real": [], "tts": []}
    for w in wavs:
        name = os.path.basename(w)
        feats = {k: v for k, v in extract_features(w).items() if k in KEYS}
        groups["real" if name.startswith("real_") else "tts"].append((name, feats))

    print(f"{'feature':16s} {'real: mean':>12s} {'tts: mean':>12s}   10-90pct real / tts -> separation")
    for k in KEYS:
        rv = np.array([f[k] for _, f in groups["real"]], dtype=float)
        tv = np.array([f[k] for _, f in groups["tts"]], dtype=float)
        rv, tv = rv[~np.isnan(rv)], tv[~np.isnan(tv)]
        if len(rv) < 3 or len(tv) < 3:
            print(f"{k:16s} {'insufficient data':>12s} (real n={len(rv)}, tts n={len(tv)})")
            continue
        hi_real, lo_real = np.percentile(rv, [90, 10])
        hi_tts, lo_tts = np.percentile(tv, [90, 10])
        sep = "SEPARATES" if (lo_real > hi_tts or lo_tts > hi_real) else "overlaps"
        print(f"{k:16s} {rv.mean():12.4f} {tv.mean():12.4f}   "
              f"[{lo_real:.4f},{hi_real:.4f}] / [{lo_tts:.4f},{hi_tts:.4f}] -> {sep}")

    print("\nper-clip detail:")
    for g in ("real", "tts"):
        for name, f in groups[g]:
            print(f"  {g}  {name:30s} " + "  ".join(f"{k}={f[k]}" for k in KEYS))


if __name__ == "__main__":
    main()
