"""
Combines multiple detection signals into a single 0-100 risk score.

Weights kept at spoof 0.75 / mismatch 0.25 after the 2026-09-06 stress test
(30 clips, samples/PROVENANCE-stress.md + RESULTS.md) -- the retune that day
was in models/anti_spoof.py (two-model ensemble), not here. Numbers that
pin these constants with the ensemble feeding spoof_risk:
  - ECAPA similarity: same speaker ~0.92-0.95 clean, stays 0.70-0.93 under
    noise/reverb/quiet. Different voice ~0.00-0.18. A synthetic voice is
    just as self-consistent as a real one (two clips of the same TTS voice
    scored 0.947), so speaker verification CANNOT catch a good clone of the
    enrolled person -- the anti-spoof signal is the only one that can.
    Hence spoof gets 3x the weight.
  - Measured with these weights (spoof 0.75 / mismatch 0.25):
      trusted person live:            ~10/100 (no alert)
      + white noise at 5 dB SNR:      ~44/100 (no alert -- thinnest real margin)
      different human voice:          ~32/100 (flag voice_mismatch, no alert)
      neural TTS, other voice:        52-100   (ALERT)
  - ALERT_THRESHOLD=50 cannot move: lowering it eats the 6-pt margin under
    the white5-noise real case (43.8); raising it drops the weakest true
    catches (TTS at 52.2 and 53.4). 50 is measured-optimal, not a guess.
  - prosody weight is 0.0 until the P1 prosody signal actually exists;
    a nonzero weight for a signal that is always 0 would silently cap
    the maximum risk score below 100 and dilute the live signals.
"""

DEFAULT_WEIGHTS = {
    "spoof": 0.75,
    "speaker_mismatch": 0.25,
    "prosody": 0.0,  # P1: set >0 when a prosody signal is implemented
}

ALERT_THRESHOLD = 50  # 0-100; trusted-person baseline measured ~9


def compute_risk(
    spoof_risk: float,
    speaker_mismatch: float,
    prosody_anomaly: float = 0.0,
    weights: dict = None,
    alert_threshold: float = ALERT_THRESHOLD,
) -> dict:
    """
    Args:
        spoof_risk: 0-1, higher = more likely synthetic/cloned voice
        speaker_mismatch: 0-1, higher = more likely NOT the enrolled speaker
        prosody_anomaly: 0-1, higher = more unnatural rhythm/pitch (P1, optional)
    Returns:
        dict with risk_score (0-100), flags (list), alert (bool)
    """
    w = weights or DEFAULT_WEIGHTS
    raw = (
        w["spoof"] * spoof_risk
        + w["speaker_mismatch"] * speaker_mismatch
        + w["prosody"] * prosody_anomaly
    )
    risk_score = round(raw * 100, 1)

    flags = []
    # 0.5 = the calibrated AASIST decision boundary (models/anti_spoof.py):
    # at/above it the anti-spoof model says "synthetic".
    if spoof_risk >= 0.5:
        flags.append("synthesis_artifacts_detected")
    # Observed: same speaker -> mismatch 0.05-0.08, different -> 0.82-1.0,
    # so 0.6 sits in the empty zone between the two clusters.
    if speaker_mismatch > 0.6:
        flags.append("voice_mismatch")
    if prosody_anomaly > 0.6:
        flags.append("unnatural_prosody")

    return {
        "risk_score": risk_score,
        "flags": flags,
        "alert": risk_score >= alert_threshold,
    }
