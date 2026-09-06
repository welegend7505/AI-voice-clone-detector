# Project: Voice Clone Detection Backend (SIH26104)

24-hour hackathon build. AICTE problem statement: real-time detection and
prevention of voice cloning/impersonation attacks during live calls.

Full context/discussion history for this project is not in this repo —
this file is the condensed, load-bearing summary. If something here
conflicts with an assumption you're about to make, trust this file.

## Team & scope

- Member 1: frontend/product (web app, mic capture, dashboards, alert UI, demo flow)
- Member 2 (you're assisting them): backend — everything in this repo
- 2 researchers: sourcing test audio samples, validating detection accuracy,
  domain research. They are NOT writing code — don't assume test fixtures
  or sample data exist unless you see them in the repo.

### HARD BOUNDARY — backend/infra only

You are working with Member 2 ONLY. Do not write, suggest, or scaffold any
frontend code (HTML/CSS/JS UI, React components, dashboards, mic-capture
browser code, demo-flow screens) even if it would be quick to knock out.
That is Member 1's work, in a separate repo/session you don't have visibility
into. If a task seems to require frontend changes, say so explicitly and
stop at the boundary — describe what the frontend needs from the API
(already covered in the API contract below), not how to build it.

Your scope ends at: the API works correctly, predictably, and matches the
documented contract. It does NOT extend to how that data gets rendered,
styled, or interacted with on the other end.

Full-stack devs have ~1 year professional experience. Prefer straightforward,
readable implementations over clever/abstracted ones — this needs to be
debuggable at 2am by someone who isn't a specialist in audio ML.

## What this is, precisely

A demo where: a trusted person's voice is enrolled → a "call" session starts →
live audio (captured via browser mic while a call plays on speaker, NOT
actual telephony/WhatsApp integration — that's not technically accessible)
streams to this backend in ~2.5-3 sec chunks over WebSocket → each chunk is
scored for (a) anti-spoofing (is this a real human or AI-generated voice)
and (b) speaker match (does it match the enrolled trusted voice) → both
combine into a single 0-100 risk score sent back in real time.

## Priority tiers — DO NOT build P1/P2 before P0 is solid

**P0 (this repo's current state) — must work, non-negotiable:**
FastAPI + WebSocket streaming, AASIST anti-spoof scoring, ECAPA-TDNN
speaker verification, risk fusion, the 4 endpoints in `api/routes.py`.

**P1 — DONE 2026-09-06 (all three items, each verified):**
- Prosody via `praat-parselmouth`: `models/prosody.py` runs live, but
  MEASURED (prosody_explore.py, stress set) that no pitch/rhythm/quality
  feature separates real from neural TTS — everything overlaps; edge-tts
  even varies pitch MORE than the real speaker. So `prosody_anomaly`
  returns a documented 0.0 and fusion weight stays 0.0 on evidence.
  Features ARE returned per chunk as diagnostics (live F0 display etc.).
- SQLite: `storage.py` (speakers/calls/history in `sih.db`; env
  `SIH_DB_PATH` overrides). `test_persistence.py` proves a hard server
  restart mid-call loses nothing — enrollment, call, history all resume.
- Alert trigger: `alerts.py` — `[ALERT ...]` console line always; JSON
  webhook POST if env `ALERT_WEBHOOK_URL` is set (worker thread, cannot
  stall scoring; delivery verified end-to-end 2026-09-06).

**P2 — do not build unless P0 and P1 are both done with hours to spare:**
- Hindi/Hinglish ASR — if requested, use Whisper (`pip install
  openai-whisper`), NOT AI4Bharat IndicConformer (needs a custom NeMo
  fork, heavier install, gated HF access — too risky under time pressure)
- Policy engine / CRUD endpoints
- SMS/email alert delivery
- Dashboard aggregation endpoint (compute this client-side instead)

If asked to "add the full architecture" (policy engine, privacy logging
module, transcript endpoints, dashboard summary) — push back and confirm
P0 is actually working with real audio first. Scope creep here is the
single biggest risk to this project finishing on time.

## Known placeholders that MUST be fixed before the demo, not left as-is

- ~~`api/routes.py`: the placeholder `bonafide_score` → `spoof_risk`
  conversion~~ FIXED 2026-09-05, then SUPERSEDED 2026-09-06:
  `models/anti_spoof.py` is now a TWO-MODEL ENSEMBLE (AASIST +
  Wav2Vec2-Small-AntiDeepfake, combined via max of calibrated risks).
  Reason: the 2026-09-06 stress test proved AASIST alone scores clean
  modern neural TTS as bona-fide (6/8 missed, logits ABOVE real speech);
  the ensemble catches 14/14 TTS clips while keeping real speech below
  alert except under competing background voices. Evidence:
  `samples/stress/RESULTS.md`, `compare_antideepfake.py`.
- ~~`risk/fusion.py`: `DEFAULT_WEIGHTS` and `ALERT_THRESHOLD` untuned~~
  TUNED 2026-09-05 from measured scores: weights 0.75 spoof / 0.25
  speaker / 0.0 prosody (prosody signal does not exist yet; speaker
  verification provably cannot catch a clone of the enrolled voice, so
  spoof carries the alert), `ALERT_THRESHOLD` 50. Rationale in
  `risk/fusion.py` docstring. RE-VALIDATED 2026-09-06 on the stress set:
  both values measured-optimal (lowering threshold eats the white-noise
  real margin of 6 pts; raising it drops true catches at 52-53).
- Still open (updated 2026-09-06 — see samples/PROVENANCE-stress.md for
  the full limitation list with numbers):
  - ~~Genuine-clone retest~~ DONE 2026-09-06 (clone_test.py): an XTTS-v2
    clone of the enrolled speaker (12 s reference) scores spoof 0.65-0.81
    -> risk 60.8-75.0, 2/2 chunks ALERT on synthesis_artifacts, while the
    same speaker's real voice scores 19.6-32.2 with 0 alerts. Threshold 50
    and current weights hold; no retune. Caveat: an ElevenLabs-quality
    clone is untested (stock Roger scored spoof 0.39-0.77, one window near
    threshold); re-run clone_test.py if one gets built. Clone file kept at
    ../my_clone.wav (outside the repo; copy in if wanted as a demo asset).
  - Competing background speech (another voice within ~20 dB of the
    caller) false-alerts on real speech — both detectors and ECAPA
    degrade together. Keep the demo mic away from other voices.
  - AASIST's ensemble threshold (−1.40) sits in a 0.22-logit gap fitted
    on one clip per side; fragile until re-verified on team recordings.
  - Real-voice base samples are still 8 kHz telephone speech, not true
    16 kHz mic audio.

## Environment notes

- Both ML models (AASIST, SpeechBrain ECAPA-TDNN) download from
  huggingface.co on first run — requires real internet access, won't
  work in a fully offline/sandboxed environment.
- AASIST runs via the repo's `aasist.onnx` + onnxruntime (added to
  requirements.txt). The HF repo's README shows a python wrapper
  (`from aasist import AASIST`) but that file does not exist in the repo.
- Windows quirks (both hit and fixed 2026-09-05, don't reintroduce):
  speechbrain needs `local_strategy=LocalStrategy.COPY` (symlinks need
  Developer Mode), and `verify_files()` mangles absolute Windows paths —
  `models/speaker.py` loads audio itself and calls `verify_batch()`
  with tensors instead.
- venv lives at `sih-backend/venv` (Python 3.14). Server:
  `venv\Scripts\python.exe -m uvicorn main:app --port 8000`.
- Target audio format throughout: 16kHz mono WAV. `streaming/audio_utils.py`
  handles resampling from whatever the frontend sends.
- WebSocket message format from frontend: `{"audio_hex": "<hex-encoded
  wav bytes>"}`. If the frontend dev (Member 1) changes this framing,
  update `api/routes.py`'s `stream_audio` function to match — don't
  silently assume it's still hex-encoded.

## Before touching this repo further

Run `test_models.py` against real audio samples if it hasn't been run
yet this session. Don't build on top of unverified model behavior.

Phase-2 stress-test tooling (2026-09-06, all rerunnable):
- `make_stress_samples.py` — regenerates samples/stress/ (seeded)
- `stress_test.py` — full-pipeline scores for every stress clip → RESULTS.md
- `compare_antideepfake.py` — AASIST vs W2V2 side-by-side evidence
- `api_smoke_test.py` — live enroll→call→WS→risk check (needs server running)
- `prosody_explore.py` — re-measures prosody non-separation via the
  production extractor in models/prosody.py
- `test_persistence.py` — server-restart survival check (own server :8001)
- `test_hardening.py` — 24 failure-scenario checks (own server :8002):
  disconnect+reconnect, silent/empty/click/malformed/truncated chunks,
  concurrent calls (no enrollment leak), REAL model failure mid-stream
  (corrupted enrollment file) -> result-shaped
  "detection_failed_this_chunk" fallback, session survives and recovers.
  Added 2026-09-06 with the hardening itself: speech-presence gate
  (MIN_VOICED_FRAC in api/routes.py -- click+silence scored as FAKE
  before: spoof 0.565 -> false alert; pitch voiced_frac separates
  click 0.00 from speech 0.31+), scoring moved to asyncio.to_thread
  (concurrent calls no longer serialize: 2 streams, 6 chunks, 4.0s wall).
- `calibrate.py` still runs but prints AASIST-only suggestions from the
  LEGACY samples/fake SAPI set — do not paste them into the ensemble's
  CALIBRATION dict; per-model constants there are the tuned source of truth.

See README.md for full setup commands and the complete API contract.
