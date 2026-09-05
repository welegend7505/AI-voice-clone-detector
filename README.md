# Voice Clone Detection Backend — SIH26104

P0-scoped backend skeleton: the core pipeline only. Policy engine, ASR,
prosody, alerts, and dashboard aggregation are intentionally left out —
see "What's NOT here" below. See CLAUDE.md for the full project context
that Claude Code reads automatically.

## 0. Do this first (30 min, before writing any more code)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python test_models.py sample1.wav sample2.wav
```

This downloads both models (AASIST + SpeechBrain ECAPA-TDNN) from Hugging
Face and confirms they load and run on your machine. If this fails, fix
it now — don't discover a broken dependency at hour 10 while wiring the
full pipeline.

Test with:
- Two recordings of the same real person → expect high speaker similarity
- A real sample + an AI-cloned sample of the same voice (make one with any
  free TTS tool) → note how the AASIST bonafide_score shifts

Status (2026-09-05): pipeline verified end-to-end with local samples
(`samples/` + `samples/PROVENANCE.md`). The AASIST side now runs the
repo's `aasist.onnx` via onnxruntime — the python wrapper its README
mentions does not actually exist in the repo.

**Recalibrating after getting real demo audio** (do this once the
researchers provide a genuine recording + an AI clone of the same voice):

```bash
# put real recordings in samples/real/, clones in samples/fake/
python calibrate.py          # prints suggested CALIBRATION constants
# paste them into models/anti_spoof.py CALIBRATION
```

Current constants are PRELIMINARY (see the caveats in
`samples/PROVENANCE.md`): calibrated against 8 kHz telephone speech and
classic Windows SAPI TTS, not 16 kHz mic audio and a modern neural clone.

**Full API loop test** (server must be running):

```bash
python test_api.py            # 23 checks: contract shape, error handling,
                              # reconnect survival; exits non-zero on failure
python test_persistence.py    # 3 checks: hard server restart mid-call loses
                              # nothing (starts its own server on :8001)
python test_hardening.py      # 24 checks: disconnects, silent/click/malformed
                              # chunks, concurrent calls, model failure
                              # mid-stream (own server on :8002)
```

## 1. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/health` → should return `{"status": "ok"}`.

## 2. API contract (share with Member 1 / frontend now)

**`POST /api/v1/speakers/enroll`**
Multipart file upload (15-20 sec clean voice sample) → `{"speaker_id": "..."}`

**`POST /api/v1/calls?speaker_id=...`**
→ `{"call_id": "..."}`

**`WS /api/v1/calls/{call_id}/stream`**
Client sends every ~2.5-3 sec:
```json
{"audio_hex": "<hex-encoded wav bytes>"}
```
Server responds per chunk:
```json
{
  "call_id": "...",
  "risk_score": 0-100,
  "flags": ["synthesis_artifacts_detected", "voice_mismatch"],
  "alert": true,
  "spoof_risk": 0.0-1.0,
  "speaker_mismatch": 0.0-1.0
}
```
Extra additive fields beyond the contract core (ignore if unneeded):
`aasist_logit_raw` / `w2v2_logit_raw` (the two anti-spoof models' raw
logits, for debugging why spoof_risk has its value), `prosody_anomaly`
(always 0.0 for now — measured not to discriminate, see
`models/prosody.py`), and `prosody_features` (live pitch/rhythm stats —
mean_f0, f0_cv, voiced_frac, jitter_local, syll_nuclei_rate, pause_frac —
free material for a live voice-analysis display in the UI).

Bad input (non-JSON message, missing/invalid `audio_hex`, unparseable,
too-short, silent, or speech-free audio — e.g. a mic bump) gets
`{"error": "<reason>"}` on the same socket — the connection stays open
and the next chunk proceeds normally.
If the MODELS themselves fail on a chunk, the reply is result-shaped
with nulls instead of a stack trace: `{"call_id", "risk_score": null,
"spoof_risk": null, "speaker_mismatch": null, "flags": [], "alert":
false, "error": "detection_failed_this_chunk", "detail": "..."}` — the
call continues and `GET /risk` keeps serving the last good score.
A dropped connection does not end the call: reconnect to the same
`call_id` and keep streaming. Concurrent calls are scored in parallel
threads and never share state.

**`GET /api/v1/calls/{call_id}/risk`**
Returns the most recent risk result for that call.

**Confirm with Member 1 before either of you builds further:** exact
chunk size, hex vs. base64 encoding, and sample rate. A mismatch here is
the classic "two halves don't connect" bug.

## 3. Priority tiers (build in this order)

**P0 — this skeleton, must work:**
FastAPI + WebSocket streaming, AASIST anti-spoof, ECAPA-TDNN speaker
verification, risk fusion.

**P1 — DONE 2026-09-06 (all three):**
- ~~Prosody signal~~ built & measured: `models/prosody.py` extracts
  pitch/rhythm features live, but NO feature separated real speech from
  neural TTS on the stress set, so `prosody_anomaly` returns 0.0 and the
  fusion weight stays 0.0 on evidence (`prosody_explore.py` reproduces)
- ~~SQLite~~ `storage.py` (speakers/calls/history; DB file `sih.db`,
  override with `SIH_DB_PATH`). Verified: enroll + stream, hard-kill the
  server, restart, continue the SAME call (`test_persistence.py`)
- ~~Alert trigger~~ `alerts.py`: greppable `[ALERT ...]` console line on
  every alerting chunk + optional JSON webhook POST if env
  `ALERT_WEBHOOK_URL` is set (fire-and-forget, never blocks scoring)

**P2 — cut without guilt if short on time:**
- Hindi/Hinglish ASR (Whisper is a far lighter dependency than
  AI4Bharat's IndicConformer, which needs a custom NeMo fork — skip
  IndicConformer under time pressure)
- Configurable policy engine / CRUD endpoints
- SMS/email alert delivery
- Dashboard aggregation endpoint (compute this client-side instead)

## 4. What's NOT here (by design)

`policies/`, `privacy/`, SMS/email delivery, and most of the REST surface
from the original architecture diagram are deliberately omitted. Add them
only after `test_models.py` passes and a full enroll → call → stream →
risk round trip works with real audio. Show the fuller architecture in
your pitch deck as the designed system — just don't claim in the live
demo that unbuilt parts are working.
