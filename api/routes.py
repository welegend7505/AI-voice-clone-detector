"""
P0 API routes only -- the non-negotiable core:
  POST /api/v1/speakers/enroll
  POST /api/v1/calls
  WS   /api/v1/calls/{call_id}/stream
  GET  /api/v1/calls/{call_id}/risk

Policy CRUD, dashboard summary, transcript endpoints, alerts list, etc.
are intentionally NOT here -- add them only if P0 is solid with time to
spare (see README priority tiers).

Storage is SQLite (storage.py, P1 2026-09-06): enrollments, calls, and
per-chunk risk history survive a server restart mid-demo.

RESPONSE CONTRACT (Member 1 builds against this -- do not change):
Each processed WebSocket chunk returns exactly:
  {"call_id", "risk_score" (0-100), "flags" (list), "alert" (bool),
   "spoof_risk" (0-1), "speaker_mismatch" (0-1)}
Bad input gets {"error": "<human-readable reason>"} on the same socket
and the connection stays open so the caller can retry.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import alerts
import storage
from models import anti_spoof, prosody, speaker
from risk.fusion import compute_risk
from streaming.audio_utils import bytes_to_wav_file, chunk_stats, probe_audio

router = APIRouter()

MIN_CHUNK_SECONDS = 0.5   # shorter chunks give the models nothing to work with
MIN_ENROLL_SECONDS = 2.0  # a useful voiceprint needs at least this much speech
SILENCE_RMS = 0.005       # below this a chunk is effectively silence
MIN_VOICED_FRAC = 0.05    # pitch-tracker voiced fraction: below this the
                          # chunk has no speech (a transient/click passes the
                          # RMS check but scores as fake -- measured
                          # 2026-09-06: click 0.00 vs speech 0.31+)


def _log_intake(tag: str, src: str, audio_bytes: bytes,
                rms: float | None = None) -> None:
    """One console line per incoming recording/chunk showing the ACTUAL
    pre-normalization format. Integration-debug surface: if the frontend
    sends something other than agreed (48kHz, stereo, webm instead of
    WAV, 0.4s chunks), it shows here immediately instead of surfacing as
    an opaque error on the frontend side."""
    info = probe_audio(audio_bytes)
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    dur = f"{info['duration_s']:.2f}" if info["duration_s"] is not None else "?"
    rate = f"{info['sr']}Hz x{info['channels']}ch" if info["sr"] else "UNREADABLE"
    level = f" rms={rms:.4f}" if rms is not None else ""
    print(f"[{tag} {ts}] {src} {dur}s {rate} {info['format']}{level} "
          f"({len(audio_bytes)}B)", flush=True)


@router.post("/api/v1/speakers/enroll")
async def enroll_speaker(file: UploadFile = File(...)):
    """Upload a 15-20 sec clean sample of the trusted person's voice."""
    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"error": "empty file upload"})

    try:
        wav_path = bytes_to_wav_file(audio_bytes)
        stats = chunk_stats(wav_path)
    except Exception:
        _log_intake("ENROLL-REJ", "upload", audio_bytes)
        return JSONResponse(
            status_code=400,
            content={"error": "upload is not a readable audio file (send WAV/PCM)"},
        )
    _log_intake("ENROLL", "upload", audio_bytes, rms=stats["rms"])

    if stats["duration_s"] < MIN_ENROLL_SECONDS:
        return JSONResponse(
            status_code=400,
            content={"error": f"sample too short ({stats['duration_s']:.1f}s); "
                              f"need at least {MIN_ENROLL_SECONDS:.0f}s of speech"},
        )
    if stats["rms"] < SILENCE_RMS:
        return JSONResponse(status_code=400, content={"error": "sample is silent"})

    # Store the NORMALIZED (16 kHz mono) wav we just wrote, then drop the temp file.
    with open(wav_path, "rb") as f:
        normalized_bytes = f.read()
    os.remove(wav_path)
    speaker_id = storage.save_speaker(normalized_bytes)
    return {"speaker_id": speaker_id}


@router.post("/api/v1/calls")
async def create_call(speaker_id: str):
    if not storage.speaker_exists(speaker_id):
        return JSONResponse(status_code=404, content={"error": "unknown speaker_id"})
    return {"call_id": storage.create_call(speaker_id)}


@router.get("/api/v1/calls/{call_id}/risk")
async def get_latest_risk(call_id: str):
    if not storage.get_call(call_id):
        return JSONResponse(status_code=404, content={"error": "unknown call_id"})
    result = storage.latest_risk(call_id)
    if result is None:
        return JSONResponse(status_code=404, content={"error": "no data yet"})
    return result


def _score_chunk(chunk_wav: str, enrolled_wav: str, pros: dict) -> dict:
    """Run both models on one chunk and fuse. Raises whatever the models raise.
    `pros` is the prosody result computed during validation (see handler)."""
    scored = anti_spoof.score_audio(chunk_wav)
    spoof_risk = scored["spoof_risk"]  # ensemble: max over available models

    similarity = speaker.verify_speaker(enrolled_wav, chunk_wav)["similarity_score"]
    # Cosine similarity ranges [-1, 1] and can land slightly negative for
    # very different voices -- clamp so the documented 0.0-1.0 range holds.
    speaker_mismatch = min(1.0, max(0.0, 1 - similarity))

    # P1 prosody signal: measured to NOT separate real from neural TTS
    # (models/prosody.py docstring), so anomaly is a documented 0.0 and
    # fusion ignores it -- the features below are diagnostics only.
    risk = compute_risk(spoof_risk=spoof_risk, speaker_mismatch=speaker_mismatch,
                        prosody_anomaly=pros["prosody_anomaly"])
    return {
        "risk_score": risk["risk_score"],
        "flags": risk["flags"],
        "alert": risk["alert"],
        "spoof_risk": round(spoof_risk, 3),
        "speaker_mismatch": round(speaker_mismatch, 3),
        "prosody_anomaly": round(pros["prosody_anomaly"], 3),
        # Extra debug fields (additive, not in README contract): the raw
        # per-model logits, so you can see WHY spoof_risk has its value.
        # (None/null = that model fell back / could not load.)
        "aasist_logit_raw": (round(scored["aasist_logit"], 3)
                             if scored["aasist_logit"] is not None else None),
        "w2v2_logit_raw": (round(scored["w2v2_logit"], 3)
                           if scored["w2v2_logit"] is not None else None),
        # Diagnostics for optional frontend display (pitch/rhythm stats).
        "prosody_features": pros["features"],
    }


@router.websocket("/api/v1/calls/{call_id}/stream")
async def stream_audio(websocket: WebSocket, call_id: str):
    """
    Frontend sends JSON text messages: {"audio_hex": "<hex-encoded wav bytes>"}
    every ~2.5-3 sec. Server responds with a risk-score JSON after each chunk.

    Bad input (not JSON, wrong field, bad hex, unparseable/too-short/silent/
    speech-free audio) gets an {"error": ...} reply and the socket STAYS OPEN.
    The server never crashes or silently drops the connection on bad input.

    If the MODELS raise on a chunk (corrupt state, disk problem, ...), the
    reply is result-shaped with nulls instead of a stack trace:
      {"call_id", "risk_score": null, "spoof_risk": null,
       "speaker_mismatch": null, "flags": [], "alert": false,
       "error": "detection_failed_this_chunk", "detail": "..."}
    and the call continues; GET /risk keeps serving the last GOOD score.
    """
    await websocket.accept()
    call = storage.get_call(call_id)
    if not call:
        await websocket.send_json({"error": "invalid call_id"})
        await websocket.close(code=1008)  # policy violation: unknown call
        return

    enrolled_wav = storage.get_speaker_wav_path(call["speaker_id"])

    try:
        while True:
            # Use raw receive() instead of receive_text(): a binary frame
            # has no "text" key and receive_text() would raise KeyError
            # (killing the socket with a 1011 instead of a clean error).
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            raw = message.get("text")
            if raw is None:
                await websocket.send_json({
                    "error": "expected a JSON text message, "
                             "got a binary WebSocket frame"})
                continue

            # --- validate the message itself ---
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "message is not valid JSON; "
                             "expected {\"audio_hex\": \"<hex-encoded wav bytes>\"}"})
                continue

            if not isinstance(payload, dict) or not isinstance(payload.get("audio_hex"), str):
                await websocket.send_json({
                    "error": "missing or non-string 'audio_hex' field; "
                             "expected {\"audio_hex\": \"<hex-encoded wav bytes>\"}"})
                continue

            try:
                audio_bytes = bytes.fromhex(payload["audio_hex"])
            except ValueError:
                await websocket.send_json({"error": "audio_hex is not valid hex"})
                continue

            # --- validate the audio itself ---
            try:
                chunk_wav = bytes_to_wav_file(audio_bytes)
            except Exception:
                _log_intake("CHUNK-REJ", call_id[:8], audio_bytes)
                await websocket.send_json({
                    "error": "audio chunk is not a readable audio file (send WAV/PCM)"})
                continue

            try:
                stats = chunk_stats(chunk_wav)
                # Logged BEFORE the checks below, so rejected chunks (too
                # short, silent, no speech) show their numbers here too.
                _log_intake("CHUNK", call_id[:8], audio_bytes, rms=stats["rms"])
                if stats["duration_s"] < MIN_CHUNK_SECONDS:
                    await websocket.send_json({
                        "error": f"audio chunk too short ({stats['duration_s']:.2f}s); "
                                 f"need at least {MIN_CHUNK_SECONDS}s"})
                    continue
                if stats["rms"] < SILENCE_RMS:
                    await websocket.send_json({
                        "error": "audio chunk is silent; nothing to score"})
                    continue

                # Speech-presence gate (also produces this chunk's prosody
                # diagnostics, reused below -- extracted once per chunk):
                # a transient (mic bump / click) passes the RMS check but
                # has no pitch, and the models score it as borderline
                # FAKE (measured: click+silence -> spoof_risk 0.565 ->
                # false ALERT). Pitch voiced fraction separates cleanly:
                # click 0.00 vs quietest real/TTS speech 0.31+.
                pros = prosody.score_audio(chunk_wav)
                voiced = pros["features"].get("voiced_frac") or 0.0
                if voiced < MIN_VOICED_FRAC:
                    await websocket.send_json({
                        "error": "no speech detected in chunk "
                                 "(transient/click only); nothing to score"})
                    continue

                # --- score + respond ---
                # Model inference runs in a worker thread so a second
                # concurrent call's socket I/O is not blocked for ~1 s.
                scored = await asyncio.to_thread(
                    _score_chunk, chunk_wav, enrolled_wav, pros)
                result = {"call_id": call_id, **scored}
                storage.append_history(call_id, result)
                if result["alert"]:
                    # Webhook POST happens in a worker thread; a slow/dead
                    # endpoint must never stall this scoring loop.
                    asyncio.get_running_loop().run_in_executor(
                        None, alerts.fire_alert, result)
                await websocket.send_json(result)
            except Exception as e:
                # A model/scoring failure on ONE chunk must not kill the
                # call. Send a result-SHAPED fallback (risk_score null =
                # "no data this chunk"; the frontend can treat it exactly
                # like an error reply) and keep the socket open.
                # Deliberately NOT appended to history: GET /risk keeps
                # returning the last GOOD score instead of a null row.
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[CHUNK-ERR {ts}] call={call_id[:8]} scoring failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                await websocket.send_json({
                    "call_id": call_id,
                    "risk_score": None,
                    "spoof_risk": None,
                    "speaker_mismatch": None,
                    "flags": [],
                    "alert": False,
                    "error": "detection_failed_this_chunk",
                    "detail": f"{type(e).__name__}: {e}"})
            finally:
                # Temp chunks pile up fast (one per ~3s); enrolled wavs must stay.
                try:
                    os.remove(chunk_wav)
                except OSError:
                    pass

    except WebSocketDisconnect:
        # Session state lives in SQLite, so the client can just open a new
        # WebSocket to the same call_id and continue where it left off --
        # even if the server was restarted in between.
        pass
