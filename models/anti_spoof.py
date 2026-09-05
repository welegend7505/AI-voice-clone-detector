"""
Anti-spoofing / voice-clone detection -- TWO-MODEL ENSEMBLE (2026-09-06).

Model 1: AASIST  (SpeechAntiSpoofingBenchmarks/AASIST, MIT)
Model 2: Wav2Vec2-Small-AntiDeepfake (SpeechAntiSpoofingBenchmarks/
         Wav2Vec2-Small-AntiDeepfake, CC BY-NC-SA -- fine for a
         non-commercial hackathon demo, NOT for a commercial product)

WHY TWO MODELS (measured 2026-09-06 on the samples/stress set, 30 clips
of real speech under 12 degradation conditions + edge-tts neural TTS):
  - AASIST is BLIND to clean modern neural TTS: 6 of 8 clean neural clips
    scored as bona-fide, often MORE bona-fide than the real human
    (e.g. tts_US-Jenny_1 logit +4.75 vs real_clean +0.90). It does catch
    TTS that has been through noise/bandlimiting/reverb (6 of 6).
  - W2V2-AntiDeepfake catches clean neural TTS (6 of 8; 2 marginal) and is
    very robust for real speech under noise (real + white 5 dB: +3.15 vs
    AASIST -1.30), but noise MASKS its cues on synthetic audio (0 of 5
    degraded TTS caught) and a competing background voice at <=15 dB SNR
    makes it call real speech synthetic.
  - max(risk_aasist, risk_w2v2) catches 14/14 TTS clips in our set while
    keeping every real clip below alert except competing-speech babble
    (a documented known limitation -- see samples/PROVENANCE-stress.md).

Per-model ONNX protocols (both verified empirically):
  AASIST : input "wav" [batch, 64600] -> "logits" [batch, 2];
           index 1 = bona-fide.  (4.04 s window)
  W2V2   : input "wav" [batch, 64000] -> "logits" [batch, 2];
           index 1 = real.  (4.00 s window; near-symmetric +/- logits)
Both: raw 16 kHz mono float32, cropped to the first window / zero-padded.

Requires internet on first run (huggingface_hub snapshot download).
If one model fails to load, scoring falls back to the other and the
result dict says so via "models_used" -- the demo keeps running.
"""

import os

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import snapshot_download

MODELS = {
    "aasist": {
        "repo_id": "SpeechAntiSpoofingBenchmarks/AASIST",
        "onnx_file": "aasist.onnx",
        "window": 64600,
        "real_logit_index": 1,
    },
    "w2v2": {
        "repo_id": "SpeechAntiSpoofingBenchmarks/Wav2Vec2-Small-AntiDeepfake",
        "onnx_file": "wav2vec2-small-antideepfake.onnx",
        "window": 64000,
        "real_logit_index": 1,
    },
}

# ---------------------------------------------------------------------------
# Calibration: raw real-logit -> 0-1 spoof risk, ONE entry per model.
#
# Tuned 2026-09-06 from samples/stress (30 clips; see PROVENANCE-stress.md):
#   w2v2   threshold 0.0 = the model's own trained decision boundary -- our
#          real-speech minimum (excl. competing-voice babble) is +1.50, so
#          every real clip lands at risk <= 0.25.
#   aasist threshold -1.40 sits between the noisiest real clip
#          (real_white5 dB, logit -1.30) and the weakest degraded-TTS clip
#          (tts_US-Guy_1_phonespk, -1.52). THIN margin (0.2 logits) and
#          fitted on ONE clip per side -- treat as fragile, re-check with
#          the team's own recordings before the demo.
# Combined spoof_risk = max over models (either detector firing is enough).
# ---------------------------------------------------------------------------
CALIBRATION = {
    "aasist": {"threshold": -1.40, "ramp_halfwidth": 3.0},
    "w2v2": {"threshold": 0.0, "ramp_halfwidth": 3.0},
}

_sessions = {}  # model key -> ort.InferenceSession (or None if unloadable)


def _get_session(key: str):
    """Lazy-load one ONNX session; caches None on failure so we don't retry."""
    if key not in _sessions:
        try:
            spec = MODELS[key]
            repo_path = snapshot_download(repo_id=spec["repo_id"])
            onnx_path = os.path.join(repo_path, spec["onnx_file"])
            _sessions[key] = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])
        except Exception:
            _sessions[key] = None  # offline / repo gone -> fall back to the other
    return _sessions[key]


def _load_as_model_input(wav_path: str, window: int) -> np.ndarray:
    """Any wav -> float32 mono 16 kHz, exactly `window` samples, batched [1, n]."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse stereo to mono
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    data = np.asarray(data, dtype=np.float32)
    if len(data) < window:
        data = np.pad(data, (0, window - len(data)))  # zero-pad short chunks
    else:
        data = data[:window]  # deterministic first-window crop
    return np.ascontiguousarray(data)[None, :]


def _real_logit(key: str, wav_path: str):
    """One model's real-logit for a wav, or None if that model is unavailable."""
    session = _get_session(key)
    if session is None:
        return None
    spec = MODELS[key]
    x = _load_as_model_input(wav_path, spec["window"])
    logits = session.run(None, {"wav": x})[0][0]
    return float(logits[spec["real_logit_index"]])


def score_audio(wav_path: str) -> dict:
    """
    Args:
        wav_path: path to a wav file (any sample rate soundfile can read;
                  it gets resampled to 16 kHz mono per model)
    Returns:
        dict with:
          spoof_risk       -- combined 0-1 risk: max over available models
          aasist_logit     -- raw AASIST real-logit (None if model unavailable)
          w2v2_logit       -- raw W2V2 real-logit (None if model unavailable)
          bonafide_score   -- ALIAS of aasist_logit, kept so older scripts
                              (test_models.py, calibrate.py) keep working
          models_used      -- which models contributed
        Raises RuntimeError if BOTH models are unavailable.
    """
    logits = {k: _real_logit(k, wav_path) for k in MODELS}
    used = [k for k, v in logits.items() if v is not None]
    if not used:
        raise RuntimeError("no anti-spoof model could be loaded (offline?)")

    risks = [spoof_risk_from_logit(logits[k], model=k) for k in used]
    return {
        "spoof_risk": max(risks),
        "aasist_logit": logits["aasist"],
        "w2v2_logit": logits["w2v2"],
        "bonafide_score": logits["aasist"],  # back-compat alias
        "models_used": used,
    }


def spoof_risk_from_logit(real_logit: float, model: str = "aasist",
                          calibration: dict = None) -> float:
    """
    Convert one model's raw real-logit to a 0-1 spoof risk.
    Linear ramp: logit >= threshold + halfwidth -> 0.0
                 logit <= threshold - halfwidth -> 1.0
    Simple on purpose -- anyone can draw this on paper at 2am.
    """
    c = (calibration or CALIBRATION)[model]
    hi = c["threshold"] + c["ramp_halfwidth"]   # risk 0.0 at/above this
    lo = c["threshold"] - c["ramp_halfwidth"]   # risk 1.0 at/below this
    if real_logit >= hi:
        return 0.0
    if real_logit <= lo:
        return 1.0
    return (hi - real_logit) / (hi - lo)


def calibrate_threshold(known_real_scores: list, known_fake_scores: list) -> float:
    """
    Quick-and-dirty threshold picker for ONE model: midpoint between the
    average real-sample logit and average fake-sample logit.
    Run against your own recordings before the demo -- do NOT trust a
    threshold from the paper, your mic/room conditions differ.
    """
    if not known_real_scores or not known_fake_scores:
        raise ValueError("Need at least one real and one fake sample score")
    avg_real = sum(known_real_scores) / len(known_real_scores)
    avg_fake = sum(known_fake_scores) / len(known_fake_scores)
    return (avg_real + avg_fake) / 2
