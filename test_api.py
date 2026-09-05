"""End-to-end API test: enroll -> create call -> WS stream -> verify contract.

Expects the server running on 127.0.0.1:8000 (uvicorn main:app --port 8000).

Covers the README contract plus the graceful-degradation cases:
  - valid chunks (real voice, then synthetic voice) -> exact response shape
  - non-JSON message / missing field / bad hex / non-audio bytes /
    too-short chunk / silent chunk -> {"error": ...} and socket stays open
  - hard disconnect + reconnect on the same call_id -> streaming continues
  - unknown speaker_id / call_id -> clean 404s, clean WS error + close

Exit code 0 = everything passed.
"""

import io
import json
import sys
import tempfile

import httpx
import soundfile as sf
import websockets.sync.client as ws_client

BASE = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"

CONTRACT_FIELDS = ["call_id", "risk_score", "flags", "alert",
                   "spoof_risk", "speaker_mismatch"]

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def wav_slice_bytes(path, seconds):
    """Read `seconds` of audio from `path`, return raw WAV file bytes."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data[: int(seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def silent_wav_bytes(seconds, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, __import__("numpy").zeros(int(seconds * sr)), sr, format="WAV")
    return buf.getvalue()


def recv_json(sock, timeout=120):
    return json.loads(sock.recv(timeout=timeout))


def main():
    real = "samples/real/OSR_us_000_0010_8k.wav"
    tts = "samples/fake/tts_david_1.wav"
    real_chunk = wav_slice_bytes(real, 3.0).hex()
    tts_chunk = wav_slice_bytes(tts, 3.0).hex()

    http = httpx.Client(base_url=BASE, timeout=120)

    # ---------- health ----------
    r = http.get("/health")
    check("GET /health", r.status_code == 200 and r.json() == {"status": "ok"},
          f"-> {r.status_code} {r.text[:80]}")

    # ---------- enroll: bad upload then good upload ----------
    r = http.post("/api/v1/speakers/enroll",
                  files={"file": ("junk.bin", b"this is not audio", "application/octet-stream")})
    check("enroll rejects non-audio", r.status_code == 400 and "error" in r.json(),
          f"-> {r.status_code} {r.text[:80]}")

    r = http.post("/api/v1/speakers/enroll",
                  files={"file": ("short.wav", wav_slice_bytes(real, 1.0), "audio/wav")})
    check("enroll rejects 1s sample", r.status_code == 400 and "error" in r.json(),
          f"-> {r.status_code} {r.text[:80]}")

    with open(real, "rb") as f:
        r = http.post("/api/v1/speakers/enroll",
                      files={"file": ("enroll.wav", f, "audio/wav")})
    ok = r.status_code == 200 and "speaker_id" in r.json()
    check("enroll real speech", ok, f"-> {r.status_code} {r.text[:80]}")
    if not ok:
        sys.exit("cannot continue without a speaker_id")
    speaker_id = r.json()["speaker_id"]

    # ---------- create call: unknown then known speaker ----------
    r = http.post("/api/v1/calls", params={"speaker_id": "does-not-exist"})
    check("create call rejects unknown speaker", r.status_code == 404, f"-> {r.status_code}")

    r = http.post("/api/v1/calls", params={"speaker_id": speaker_id})
    ok = r.status_code == 200 and "call_id" in r.json()
    check("create call", ok, f"-> {r.status_code} {r.text[:80]}")
    if not ok:
        sys.exit("cannot continue without a call_id")
    call_id = r.json()["call_id"]

    # ---------- WS: unknown call_id ----------
    sock = ws_client.connect(f"{WS_BASE}/api/v1/calls/bogus-call-id/stream")
    msg = json.loads(sock.recv(timeout=30))
    check("WS unknown call_id -> error", "error" in msg, f"-> {msg}")
    try:
        sock.close()
    except Exception:
        pass

    # ---------- WS: happy path + abuse on one socket ----------
    sock = ws_client.connect(f"{WS_BASE}/api/v1/calls/{call_id}/stream")

    import time
    t0 = time.perf_counter()
    sock.send(json.dumps({"audio_hex": real_chunk}))
    resp = recv_json(sock)
    print(f"  (first-chunk latency: {time.perf_counter() - t0:.1f}s "
          f"incl. one-time model load; subsequent chunks are faster)")
    shape_ok = all(f in resp for f in CONTRACT_FIELDS) and resp["call_id"] == call_id
    check("chunk: real voice -> contract shape", shape_ok, f"-> {resp}")
    check("chunk: real voice -> low risk",
          resp.get("risk_score", 100) < 35 and not resp.get("alert"),
          f"risk={resp.get('risk_score')} flags={resp.get('flags')}")

    sock.send(json.dumps({"audio_hex": tts_chunk}))
    resp = recv_json(sock)
    check("chunk: TTS voice -> synthesis flagged",
          "synthesis_artifacts_detected" in resp.get("flags", [])
          and resp.get("spoof_risk", 0) >= 0.5,
          f"risk={resp.get('risk_score')} spoof_risk={resp.get('spoof_risk')} flags={resp.get('flags')}")

    # --- abuse: every one of these must get an error reply and NOT kill the socket ---
    abuse = [
        ("non-JSON text", "not json at all"),
        ("JSON missing field", json.dumps({"wrong": "field"})),
        ("non-string audio_hex", json.dumps({"audio_hex": 12345})),
        ("bad hex", json.dumps({"audio_hex": "zz--not-hex--zz"})),
        ("hex of non-audio bytes", json.dumps({"audio_hex": b"not a wav file".hex()})),
    ]
    for name, message in abuse:
        sock.send(message)
        resp = recv_json(sock)
        check(f"abuse: {name} -> error, socket alive",
              "error" in resp and "risk_score" not in resp, f"-> {resp}")

    sock.send(b"\x00\x01\x02raw binary frame, not text")  # accidental binary send
    resp = recv_json(sock)
    check("abuse: binary frame -> error, socket alive", "error" in resp, f"-> {resp}")

    sock.send(json.dumps({"audio_hex": silent_wav_bytes(3.0).hex()}))
    resp = recv_json(sock)
    check("abuse: silent chunk -> error, socket alive", "error" in resp, f"-> {resp}")

    sock.send(json.dumps({"audio_hex": wav_slice_bytes(real, 0.2).hex()}))
    resp = recv_json(sock)
    check("abuse: 0.2s chunk -> error, socket alive", "error" in resp, f"-> {resp}")

    # socket must still score normally after all the abuse
    sock.send(json.dumps({"audio_hex": real_chunk}))
    resp = recv_json(sock)
    check("socket still scores after abuse", all(f in resp for f in CONTRACT_FIELDS),
          f"risk={resp.get('risk_score')}")

    # ---------- hard disconnect + reconnect on same call_id ----------
    sock.close()  # no close handshake -- server must survive
    sock = ws_client.connect(f"{WS_BASE}/api/v1/calls/{call_id}/stream")
    sock.send(json.dumps({"audio_hex": real_chunk}))
    resp = recv_json(sock)
    check("reconnect same call_id -> streaming continues",
          all(f in resp for f in CONTRACT_FIELDS), f"risk={resp.get('risk_score')}")
    sock.close()

    # ---------- GET risk ----------
    r = http.get(f"/api/v1/calls/{call_id}/risk")
    ok = r.status_code == 200 and all(f in r.json() for f in CONTRACT_FIELDS)
    check("GET latest risk", ok, f"-> {r.status_code} {r.text[:120]}")

    r = http.get("/api/v1/calls/never-existed/risk")
    check("GET risk unknown call -> 404", r.status_code == 404, f"-> {r.status_code} {r.text[:60]}")

    # ---------- server still alive after everything ----------
    r = http.get("/health")
    check("server alive at end", r.status_code == 200, f"-> {r.status_code}")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
