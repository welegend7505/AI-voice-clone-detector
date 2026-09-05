"""Demo-hardening tests: the backend must survive an imperfect live demo.

Starts its own server (port 8002, throwaway DB) and runs the five failure
scenarios end-to-end, plus a final mixed-chunk session:

  1. mid-call disconnect + reconnect on the same call_id -> resumes,
     no duplicated history rows
  2. silent / empty / speech-free (click) chunks -> clean per-chunk errors,
     models never called on garbage
  3. malformed + truncated audio_hex payloads -> clean JSON errors, never
     a stack trace
  4. two concurrent calls with DIFFERENT enrollments streaming in parallel
     -> no state leak (each scored against its own enrollment)
  5. model inference failure mid-stream (corrupted enrollment file) ->
     result-shaped detection_failed_this_chunk, socket alive, next chunk
     after the file is restored scores normally
  6. mixed session: good -> bad -> good chunks, session keeps responding

Exit code 0 = everything passed.  Usage:  python test_hardening.py
"""

import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import httpx
import numpy as np
import soundfile as sf
import websockets.sync.client as ws_client

PORT = 8002
BASE = f"http://127.0.0.1:{PORT}"
DB = os.path.abspath("_hardening_test.db")
ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = {**os.environ, "SIH_DB_PATH": DB}

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed, failed = passed + ok, failed + (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT),
         "--log-level", "error"], env=ENV, cwd=ROOT)
    for _ in range(120):
        try:
            httpx.get(f"{BASE}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    sys.exit("server did not come up")


def wav_bytes(path, seconds=3.0):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, data[: int(seconds * sr)], sr, format="WAV")
    return buf.getvalue()


def click_wav_bytes():
    """3 s of silence with one transient -- passes the RMS gate, no speech."""
    x = np.zeros(3 * 16000, dtype=np.float32)
    x[8000], x[8001] = 0.9, -0.9
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def silent_wav_bytes():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(3 * 16000), 16000, format="WAV")
    return buf.getvalue()


def send(sock, hexdata):
    sock.send(json.dumps({"audio_hex": hexdata}))
    return json.loads(sock.recv(timeout=120))


def db_history_count(call_id):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    try:
        return con.execute("SELECT COUNT(*) FROM risk_history WHERE call_id = ?",
                           (call_id,)).fetchone()[0]
    finally:
        con.close()


def enroll_and_call(http, path):
    with open(path, "rb") as f:
        sid = http.post("/api/v1/speakers/enroll",
                        files={"file": ("e.wav", f, "audio/wav")}).json()["speaker_id"]
    cid = http.post("/api/v1/calls", params={"speaker_id": sid}).json()["call_id"]
    return sid, cid


def main():
    real = os.path.join("samples", "stress", "real_clean.wav")
    enroll_real = os.path.join("samples", "stress", "real_enroll.wav")
    guy = os.path.join("samples", "stress", "tts_US-Guy_1_clean.wav")
    good = wav_bytes(real).hex()
    click = click_wav_bytes().hex()
    silent = silent_wav_bytes().hex()
    truncated = wav_bytes(real)[: int(len(wav_bytes(real)) * 0.6)].hex()

    if os.path.exists(DB):
        os.remove(DB)
    server = start_server()
    http = httpx.Client(base_url=BASE, timeout=120)

    # ---------- 1. disconnect + reconnect ----------
    sid, cid = enroll_and_call(http, enroll_real)
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid}/stream") as ws:
        r1 = send(ws, good)
    # hard close happened at the `with` exit; reconnect:
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid}/stream") as ws:
        r2 = send(ws, good)
    check("reconnect same call_id resumes scoring",
          r1.get("risk_score") is not None and r2.get("risk_score") is not None,
          f"risks {r1.get('risk_score')} -> {r2.get('risk_score')}")
    n = db_history_count(cid)
    check("no duplicated history (2 good chunks -> 2 rows)", n == 2, f"rows={n}")

    # ---------- 2. silent / empty / speech-free chunks ----------
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid}/stream") as ws:
        r = send(ws, silent)
        check("silent chunk -> clean error", "error" in r and "Traceback" not in r.get("error", ""), f"-> {r.get('error')}")
        r = send(ws, "")  # empty hex -> zero bytes
        check("empty payload -> clean error", "error" in r, f"-> {r.get('error')}")
        r = send(ws, click)
        check("click/transient chunk -> no-speech error (not scored as fake)",
              "error" in r and "speech" in r.get("error", ""), f"-> {r.get('error')}")
        r = send(ws, good)
        check("socket still scores after silent/empty/click",
              r.get("risk_score") is not None, f"risk={r.get('risk_score')}")
    n = db_history_count(cid)
    # 2 (sec 1) + 1 (the deliberate good chunk above) = 3; garbage added none
    check("garbage chunks not written to history (3 scored rows)", n == 3, f"rows={n}")

    # ---------- 3. malformed / truncated payloads ----------
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid}/stream") as ws:
        ws.send("{not json")
        r = json.loads(ws.recv(timeout=30))
        check("non-JSON -> clean error", "error" in r, f"-> {r.get('error')}")
        r = send(ws, "zz--not-hex--zz")
        check("bad hex -> clean error", "error" in r, f"-> {r.get('error')}")
        r = send(ws, b"not a wav file".hex())
        check("non-audio bytes -> clean error", "error" in r, f"-> {r.get('error')}")
        ws.send(b"\x00\x01raw binary")
        r = json.loads(ws.recv(timeout=30))
        check("binary frame -> clean error", "error" in r, f"-> {r.get('error')}")
        r = send(ws, truncated)
        check("truncated wav -> clean handling (score or error, no traceback)",
              ("risk_score" in r) or ("error" in r), f"-> risk={r.get('risk_score')} err={r.get('error')}")
        r = send(ws, good)
        check("socket still scores after malformed payloads",
              r.get("risk_score") is not None, f"risk={r.get('risk_score')}")

    # ---------- 4. two concurrent calls, different enrollments ----------
    results = {}

    def run_call(tag, enroll_path, stream_path, n_chunks):
        h = httpx.Client(base_url=BASE, timeout=120)
        s, c = enroll_and_call(h, enroll_path)
        with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{c}/stream") as ws:
            rs = [send(ws, wav_bytes(stream_path).hex()) for _ in range(n_chunks)]
        results[tag] = {"call_id": c, "risks": rs}

    tA = threading.Thread(target=run_call, args=("A", enroll_real, real, 3))
    tB = threading.Thread(target=run_call, args=("B", guy, real, 3))
    tA.start(); tB.start(); tA.join(); tB.join()

    a_ok = all(r.get("risk_score") is not None for r in results["A"]["risks"])
    b_ok = all(r.get("risk_score") is not None for r in results["B"]["risks"])
    check("concurrent calls both keep scoring", a_ok and b_ok)
    check("calls are distinct objects",
          results["A"]["call_id"] != results["B"]["call_id"])
    a_mism = sum(r["speaker_mismatch"] for r in results["A"]["risks"]) / 3
    b_mism = sum(r["speaker_mismatch"] for r in results["B"]["risks"]) / 3
    # A: real voice vs ITS OWN enrollment -> low mismatch.
    # B: SAME real voice vs the OTHER enrollment (Guy TTS) -> high mismatch.
    # If state leaked, B would also show low mismatch.
    check("no enrollment leak: A mism low, B mism high",
          a_mism < 0.35 and b_mism > 0.6, f"A={a_mism:.3f} B={b_mism:.3f}")

    # ---------- 5. model failure mid-stream (corrupted enrollment file) ----------
    cache = os.path.join(ROOT, "enrolled_speakers", f"{sid}.wav")
    with open(enroll_real, "rb") as f:
        original = f.read()
    with open(cache, "wb") as f:
        f.write(b"corrupted-not-a-wav")
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid}/stream") as ws:
        r = send(ws, good)
        check("model failure -> result-shaped fallback",
              r.get("error") == "detection_failed_this_chunk"
              and r.get("risk_score") is None and r.get("alert") is False,
              f"-> risk={r.get('risk_score')} err={r.get('error')}")
        r = send(ws, good)
        check("socket ALIVE after model failure (second failure also clean)",
              r.get("error") == "detection_failed_this_chunk", f"-> {r.get('error')}")
    r = httpx.get(f"{BASE}/api/v1/calls/{cid}/risk", timeout=30).json()
    check("GET risk during failure serves last GOOD score",
          r.get("risk_score") is not None, f"-> {r.get('risk_score')}")
    n = db_history_count(cid)
    # 3 (through sec 2) + 2 (sec 3: truncated scored + good) = 5;
    # the two FAILED chunks above added none
    check("failed chunks not in history (5 scored rows)", n == 5, f"rows={n}")
    with open(cache, "wb") as f:
        f.write(original)  # restore
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid}/stream") as ws:
        r = send(ws, good)
        check("recovers once the file is restored",
              r.get("risk_score") is not None, f"risk={r.get('risk_score')}")

    # ---------- 6. final mixed session (the user-facing run-through) ----------
    _, cid2 = enroll_and_call(http, enroll_real)
    tts_chunk = wav_bytes(guy).hex()
    sequence = [("good real", good, "score"), ("bad hex", "xyz", "error"),
                ("click", click, "error"), ("good real", good, "score"),
                ("silent", silent, "error"), ("tts", tts_chunk, "score"),
                ("truncated", truncated, "either"), ("good real", good, "score")]
    survived = True
    with ws_client.connect(f"ws://127.0.0.1:{PORT}/api/v1/calls/{cid2}/stream") as ws:
        for name, chunk, want in sequence:
            r = send(ws, chunk)
            got = ("score" if r.get("risk_score") is not None
                   else "error" if "error" in r else "?")
            ok = (got == want) or (want == "either" and got in ("score", "error"))
            survived = survived and ok and got != "?"
            print(f"    mixed: {name:10s} -> {got:5s} "
                  f"risk={r.get('risk_score')} alert={r.get('alert')} {'ok' if ok else 'BAD'}")
    check("mixed good/bad chunk session survives end-to-end", survived)
    r = httpx.get(f"{BASE}/api/v1/calls/{cid2}/risk", timeout=30).json()
    check("GET risk after mixed session -> last good score",
          r.get("risk_score") is not None, f"-> {r.get('risk_score')}")
    n = db_history_count(cid2)
    check("mixed session history = scored chunks only", n == 5, f"rows={n}")

    server.kill()
    server.wait(timeout=10)
    for _ in range(10):
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
