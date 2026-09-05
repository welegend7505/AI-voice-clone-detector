"""Side-by-side: current AASIST vs candidate Wav2Vec2-Small-AntiDeepfake.

Scores every samples/stress/*.wav (except the enrollment reference) with both
anti-spoof models and prints each one's REAL logit (higher = more genuine),
so we can decide a model swap on the full matrix rather than a probe subset.

Usage:  python compare_antideepfake.py
"""

import glob
import os
from huggingface_hub import snapshot_download
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

from models import anti_spoof

W2V2_WINDOW = 64000  # fixed input length of the antideepfake ONNX graph


def load16k_window(path, n):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    data = np.asarray(data, dtype=np.float32)
    if len(data) < n:
        data = np.pad(data, (0, n - len(data)))
    return np.ascontiguousarray(data[:n])[None, :]


def main():
    p = snapshot_download(repo_id="SpeechAntiSpoofingBenchmarks/Wav2Vec2-Small-AntiDeepfake")
    w2v2 = ort.InferenceSession(os.path.join(p, "wav2vec2-small-antideepfake.onnx"),
                                providers=["CPUExecutionProvider"])

    wavs = sorted(w for w in glob.glob(os.path.join("samples", "stress", "*.wav"))
                  if not os.path.basename(w).startswith("real_enroll"))

    print(f"{'sample':30s} {'AASIST':>8s} {'W2V2':>8s}   truth")
    reals, fakes = [], []
    for w in wavs:
        name = os.path.basename(w)
        aasist = anti_spoof.score_audio(w)["bonafide_score"]
        w2 = float(w2v2.run(None, {"wav": load16k_window(w, W2V2_WINDOW)})[0][0][1])
        truth = "REAL " if name.startswith("real_") else "FAKE"
        print(f"{name:30s} {aasist:+8.3f} {w2:+8.3f}   {truth}")
        (reals if truth == "REAL " else fakes).append((name, aasist, w2))

    for label, vals, wrong_if in [("REAL (want high)", reals, -1), ("FAKE (want low)", fakes, 1)]:
        print(f"\n--- {label} ---")
        for name, aasist, w2 in vals:
            for model, v in [("AASIST", aasist), ("W2V2", w2)]:
                if v * wrong_if > 0:  # wrong side of 0
                    print(f"  {model} WRONG-SIDE: {name} {v:+.2f}")


if __name__ == "__main__":
    main()
