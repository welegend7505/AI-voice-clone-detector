"""
Prosody / voice-quality features via praat-parselmouth (P1, 2026-09-06).

MEASURED VERDICT -- no separation, so the fusion weight stays 0.0:
On the 30-clip stress set every candidate feature OVERLAPPED between real
speech and neural TTS (measured by prosody_explore.py, 2026-09-06):
    mean_f0      real [203,215] Hz   vs tts [144,236] Hz  -> overlaps
    f0_cv        real [0.12,0.25]    vs tts [0.21,0.31]   -> overlaps
    voiced_frac  / jitter_local / syll_rate / pause_frac  -> all overlap
Modern neural TTS has natural prosody -- edge-tts varies pitch MORE than
our real speaker does. (HNR extraction returned nothing usable at all.)
A discriminative anomaly score would need enrolled-speaker-relative
modeling, and with 1 real speaker in the test set that cannot be
validated without overfitting. Turning the weight up anyway would be an
unjustified threshold hack, so:
  - score_audio() returns prosody_anomaly = 0.0 (a documented constant)
  - the features are still returned as DIAGNOSTICS: useful for a live
    F0/rhythm display in the frontend, and as the base for a real
    prosody model if the team ever gets several real speakers + true
    clone samples to validate against.

Requires: praat-parselmouth (in requirements.txt).
"""

import parselmouth
from parselmouth.praat import call as praat_call

WINDOW_S = 4.0  # match the anti-spoof models' first-window crop


def extract_features(wav_path: str) -> dict:
    """Pitch / rhythm / voice-quality stats for the first WINDOW_S seconds.
    NaN where the clip gives the extractor nothing to measure."""
    snd = parselmouth.Sound(wav_path)
    if snd.duration > WINDOW_S:
        snd = praat_call(snd, "Extract part", 0.0, WINDOW_S,
                         "rectangular", 1.0, False)

    f = {}
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) >= 5:
        f["mean_f0"] = round(float(voiced.mean()), 1)
        f["f0_cv"] = round(float(voiced.std() / voiced.mean()), 3)
    else:
        f["mean_f0"] = f["f0_cv"] = float("nan")
    f["voiced_frac"] = round(float(len(voiced) / max(1, len(f0))), 3)

    try:
        pp = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
        f["jitter_local"] = round(float(praat_call(
            pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)), 4)
    except Exception:
        f["jitter_local"] = float("nan")

    intensity = snd.to_intensity(time_step=0.01, minimum_pitch=75)
    db = intensity.values[0]
    if db.size:
        peak_db = db.max()
        speech = db > (peak_db - 25.0)   # frames within 25 dB of the loudest
        nuclei = 0
        for i in range(1, len(db) - 1):
            if (speech[i] and db[i] >= db[i - 1] and db[i] > db[i + 1]
                    and db[i] > peak_db - 12.0):
                nuclei += 1
        f["syll_nuclei_rate"] = round(nuclei / WINDOW_S, 2)
        f["pause_frac"] = round(float(1 - speech.mean()), 3)
    else:
        f["syll_nuclei_rate"] = f["pause_frac"] = float("nan")
    return f


def score_audio(wav_path: str) -> dict:
    """
    Returns {"prosody_anomaly": 0.0, "features": {...}}.
    Feature values the extractor could not measure come back as None
    (never NaN -- NaN serializes to invalid strict JSON and would break
    the frontend's JSON.parse).
    The 0.0 is deliberate -- see the module docstring for the measured
    evidence. When a validated prosody model exists, compute the real
    score here and raise DEFAULT_WEIGHTS["prosody"] in risk/fusion.py.
    """
    features = extract_features(wav_path)
    features = {k: (None if v != v else v) for k, v in features.items()}  # NaN -> None
    return {"prosody_anomaly": 0.0, "features": features}
