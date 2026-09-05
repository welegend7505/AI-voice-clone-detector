"""SQLite persistence test: a server restart mid-call must lose nothing.

Self-contained: starts its own uvicorn on port 8001 (fresh throwaway DB),
streams a chunk, HARD-KILLS the server, restarts it, then verifies:
  1. GET risk still returns the pre-restart chunk (history survived)
  2. a new WebSocket on the SAME call_id still scores (enrollment survived)
Exit code 0 = passed.

Usage:  python test_persistence.py
"""

import io
import json
import os
import subprocess
import sys
import time

import httpx
import soundfile as sf
import websockets.sync.client as ws_client

PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
DB = os.path.abspath("_persistence_test.db")
ENV = {**os.environ, "SIH_DB_PATH": DB}

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed, failed = passed + ok, failed + (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT),
         "--log-level", "error"],
        env=ENV, cwd=os.path.dirname(os.path.abspath(__file__)))
    for _ in range(120):  # wait for readiness (first start loads 2 models)
        try:
            httpx.get(f"{BASE}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    sys.exit("server did not come up")


def wav_hex(path, seconds=3.0):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, data[: int(seconds * sr)], sr, format="WAV")
    return buf.getvalue().hex()


def main():
    real = os.path.join("samples", "stress", "real_clean.wav")
    chunk = wav_hex(real)

    if os.path.exists(DB):
        os.remove(DB)
    server = start_server()
    http = httpx.Client(base_url=BASE, timeout=120)

    with open(os.path.join("samples", "stress", "real_enroll.wav"), "rb") as f:
        speaker_id = http.post("/api/v1/speakers/enroll",
                               files={"file": ("enroll.wav", f, "audio/wav")}).json()["speaker_id"]
    call_id = http.post("/api/v1/calls", params={"speaker_id": speaker_id}).json()["call_id"]

    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{call_id}/stream") as ws:
        ws.send(json.dumps({"audio_hex": chunk}))
        before = json.loads(ws.recv(timeout=120))
    check("pre-restart chunk scored", "risk_score" in before, f"risk={before.get('risk_score')}")

    server.kill()  # hard kill, no cleanup
    server.wait(timeout=10)
    server = start_server()

    r = httpx.get(f"{BASE}/api/v1/calls/{call_id}/risk", timeout=30)
    ok = r.status_code == 200 and r.json().get("risk_score") == before["risk_score"]
    check("history survived restart (GET risk)", ok, f"-> {r.status_code} {r.text[:80]}")

    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{call_id}/stream") as ws:
        ws.send(json.dumps({"audio_hex": chunk}))
        after = json.loads(ws.recv(timeout=120))
    ok = "risk_score" in after and after["call_id"] == call_id
    check("streaming resumed on same call_id after restart", ok,
          f"risk={after.get('risk_score')}")

    server.kill()
    server.wait(timeout=10)
    for _ in range(10):  # Windows can take a moment to release the handle
        try:
            if os.path.exists(DB):
                os.remove(DB)
            break
        except PermissionError:
            time.sleep(1)
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
