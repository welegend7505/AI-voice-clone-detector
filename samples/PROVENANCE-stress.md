# Stress-test sample provenance (generated 2026-09-06, by Claude, phase 2)

**None of this is the researchers' audio or the user's own voice.** It is a
synthesized stress set built to approximate demo-day conditions. Regenerate
byte-identically with `python make_stress_samples.py` (seeded RNG, SIH26104).

## Sources

- `real_enroll.wav` / `real_clean.wav` — the two OSR Open Speech Repository
  telephone clips from `samples/real/` (same male speaker, two takes),
  resampled to 16 kHz. **Limitation carried over: the source is 8 kHz
  telephone band, not true 16 kHz mic audio** (see PROVENANCE.md).
- `samples/stress/src/*.mp3` — **edge-tts** (Microsoft Azure neural TTS,
  the `edge-tts` pip package) voices en-US-GuyNeural, en-US-JennyNeural,
  en-IN-PrabhatNeural, en-IN-NeerjaNeural, two scam-call scripts each.
  These are *modern neural synthesis* — a far closer stand-in for the demo
  adversary (ElevenLabs-class voice cloning) than the legacy SAPI samples —
  but they are **different voices, NOT clones of the enrolled speaker**.
  A true same-voice clone still needs to be generated with a real cloning
  tool (ElevenLabs / F5-TTS / XTTS) and tested before the demo.

## Degradation conditions (applied at 16 kHz, after resampling)

| suffix | condition |
|---|---|
| `white15` / `white5` | additive white noise, 15 / 5 dB SNR |
| `pink10` | pink (1/f) noise, 10 dB SNR — HVAC/fan-like |
| `babble10/15/20` | competing voice (Jenny TTS) at 10/15/20 dB SNR |
| `revsmall` / `revlarge` | synthetic room reverb, RT60 ~0.35 / 0.8 s |
| `phonespk` | 300–3400 Hz bandpass + 15 dB noise + soft clipping (call on a phone speaker) |
| `far` | −20 dB gain + small-room reverb + 25 dB pink noise |
| `gain2..50` | pure level sweep, no added noise |

## What the 2026-09-06 run found (full numbers in RESULTS.md)

1. **AASIST alone is blind to clean neural TTS** — 6/8 clean clips scored
   bona-fide, often *above* real speech (Jenny +4.75 vs real +0.90). No
   threshold fixes this: the distributions overlap.
2. This forced the **two-model ensemble** in `models/anti_spoof.py`
   (AASIST + Wav2Vec2-Small-AntiDeepfake, `max()` of calibrated risks).
3. After the swap: **14/14 TTS clips alert; 13/16 real clips stay below
   alert.** Worst real margin: white 5 dB noise at 43.8 vs threshold 50.

## KNOWN LIMITATIONS (explicit, do not "fix" with threshold hacks)

- **Competing background speech breaks the real side.** With another voice
  at 10–20 dB SNR behind the enrolled speaker, w2v2 calls the mix synthetic
  AND ECAPA mismatch degrades (0.68–0.96) → false ALERT on real speech
  (risk 51–84). Break-point is somewhere above 20 dB SNR of separation.
  Demo mitigation: mic near the call speaker, no other voices near it.
- **A clone OF THE ENROLLED VOICE under moderate noise can still slip
  through**: for a true clone the mismatch signal is ~0 (it matches), so
  only the spoof signal remains, and noise blinds w2v2 while AASIST only
  partially fires (e.g. clone + 15 dB noise ≈ risk 30 → no alert). The
  14/14 catches above lean on the mismatch signal, which a real clone
  defeats. **This is the #1 thing to retest with a genuine clone sample.**
- AASIST's new threshold (−1.40) sits in a 0.22-logit gap measured from
  ONE clip per side (real_white5 −1.30 vs tts_phonespk −1.52). Fragile —
  re-verify with team recordings.
- Neerja clean clips are the weakest catches (52–54, only ~3 pts above
  threshold) and are flagged `voice_mismatch` only, not
  `synthesis_artifacts_detected` — the w2v2 logit (+0.5) is borderline.

## Files

- `make_stress_samples.py` — regenerates everything (seeded)
- `stress_test.py` — scores all clips through the live pipeline, writes RESULTS.md
- `compare_antideepfake.py` — AASIST vs w2v2 side-by-side (model-swap evidence)
- `api_smoke_test.py` — live enroll→call→WS-stream→risk check (server must run)
