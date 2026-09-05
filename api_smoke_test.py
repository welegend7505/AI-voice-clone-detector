"""Live end-to-end smoke test of the P0 API with the ensemble anti-spoof.

Starts nothing itself -- run uvicorn first, then:
    python api_smoke_test.py
Exercises enroll -> create call -> WS stream (real chunk, noisy real chunk,
neural-TTS chunk) -> GET risk, and prints every server response verbatim.
"""

import json
import urllib.request
import uuid

import websockets

BASE = "http://127.0.0.1:8000"
CHUNKS = [
    ("real_clean.wav", False),
    ("real_white5.wav", False),
    ("real_phonespk.wav", False),
    ("tts_US-Guy_1_clean.wav", True),
    ("tts_US-Jenny_1_clean.wav", True),
]


def post_multipart(path, field, filename, content):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main():
    with open("samples/stress/real_enroll.wav", "rb") as f:
        enroll = f.read()
    speaker = post_multipart("/api/v1/speakers/enroll", "file", "enroll.wav", enroll)
    print("enroll ->", speaker)
    speaker_id = speaker["speaker_id"]

    with urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/api/v1/calls?speaker_id={speaker_id}", method="POST")) as r:
        call = json.loads(r.read())
    print("create call ->", call)
    call_id = call["call_id"]

    ok = True
    import asyncio

    async def stream():
        nonlocal ok
        async with websockets.connect(f"ws://127.0.0.1:8000/api/v1/calls/{call_id}/stream") as ws:
            for name, expect_alert in CHUNKS:
                with open(f"samples/stress/{name}", "rb") as f:
                    hexdata = f.read().hex()
                await ws.send(json.dumps({"audio_hex": hexdata}))
                reply = json.loads(await ws.recv())
                verdict = "PASS" if reply.get("alert") == expect_alert else "FAIL"
                ok = ok and verdict == "PASS"
                print(f"{name:26s} -> risk {reply.get('risk_score')!s:>5} "
                      f"spoof {reply.get('spoof_risk')} mism {reply.get('speaker_mismatch')} "
                      f"alert={reply.get('alert')} expected={expect_alert} [{verdict}]")
            # malformed message must not kill the socket
            await ws.send("not json")
            print("bad json ->", json.loads(await ws.recv()))

    asyncio.run(stream())

    with urllib.request.urlopen(f"{BASE}/api/v1/calls/{call_id}/risk") as r:
        print("GET latest risk ->", json.loads(r.read()))
    print("SMOKE TEST", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
