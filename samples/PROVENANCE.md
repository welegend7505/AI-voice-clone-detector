# Test sample provenance (generated 2026-09-05, by Claude during backend calibration)

**These are NOT the researchers' samples and NOT the user's own voice.**
They exist to exercise the pipeline and produce a *preliminary* calibration.
Before the demo, re-run `python calibrate.py` with:

- `samples/real/` containing genuine recordings of the trusted person on the
  demo mic/laptop, and
- `samples/fake/` containing an AI clone of that same person (any TTS/voice-
  clone tool), then update the constants in `models/anti_spoof.py`.

## real/
- `OSR_us_000_0010_8k.wav`, `OSR_us_000_0011_8k.wav` — real human speech,
  Open Speech Repository (voiptroubleshooter.com), US English, telephone
  band (8 kHz). Caveat: 8 kHz telephone audio has a narrower band than the
  16 kHz mic audio the demo will produce — AASIST scores may shift.

## fake/
- `tts_david_1.wav`, `tts_david_2.wav`, `tts_zira_1.wav`, `tts_zira_2.wav`
  — Windows SAPI desktop TTS voices (concatenative-ish classic voices, NOT
  a modern neural voice clone). A modern neural TTS clone is the actual
  demo adversary and will likely score differently.

## misc/
- Windows system sounds (chimes.wav, tada.wav, Windows Background.wav,
  Windows Notify.wav) — non-speech audio, used only to check the pipeline
  doesn't crash on unusual input. Scores are not meaningful either way.
