"""Scratch (2026-09-06): clean-path real-vs-TTS test, no browser involved.

Real voice  = the user's stored enrollment (enrolled_speakers/91c0d5f0...wav,
              27.7 s @ 16 kHz mono, normalized by the backend)
TTS voice   = ElevenLabs stock voice "Roger" mp3 from Downloads (27.8 s)

Both are cut into 3 x 3-second windows, enrolled ONCE from the real voice,
and streamed through the real WS API. Expected: real -> low risk / no alert;
ElevenLabs (a different speaker AND synthetic) -> high risk / alert with
both flags. This is the clean-audio counterpart of the browser test.
"""

import asyncio
import io
import json
import urllib.request
import uuid

import librosa
import soundfile as sf
import websockets

BASE = "http://127.0.0.1:8000"
REAL_WAV = "enrolled_speakers/91c0d5f0-644e-42a9-85fd-00726ec7224f.wav"
TTS_MP3 = r"C:\Users\weleg\AIclone\my_clone.wav"
WINDOW_S = 3
OFFSETS_S = [1, 8, 15]  # speech-rich windows, skipping the first second


def load_16k_mono(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    return data


def window_hexes(data, tag):
    """3 s windows -> list of (label, hex-encoded 16k mono wav bytes)."""
    out = []
    for off in OFFSETS_S:
        seg = data[off * 16000:(off + WINDOW_S) * 16000]
        buf = io.BytesIO()
        sf.write(buf, seg, 16000, format="WAV")
        out.append((f"{tag}@{off}s", buf.getvalue().hex()))
    return out


def post_multipart(path, field, filename, content):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


async def main():
    real = load_16k_mono(REAL_WAV)
    tts = load_16k_mono(TTS_MP3)
    chunks = window_hexes(real, "real") + window_hexes(tts, "tts")

    with open(REAL_WAV, "rb") as f:
        speaker = post_multipart("/api/v1/speakers/enroll", "file", "real.wav", f.read())
    with urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/api/v1/calls?speaker_id={speaker['speaker_id']}", method="POST")) as r:
        call = json.loads(r.read())
    print(f"enrolled real voice -> speaker {speaker['speaker_id'][:8]}, call {call['call_id'][:8]}\n")

    async with websockets.connect(f"ws://127.0.0.1:8000/api/v1/calls/{call['call_id']}/stream") as ws:
        results = {}
        for label, hexdata in chunks:
            await ws.send(json.dumps({"audio_hex": hexdata}))
            d = json.loads(await ws.recv())
            if d.get("error"):
                print(f"{label:10s} -> ERROR {d['error']}")
                continue
            verdict = "ALERT" if d["alert"] else "ok"
            print(f"{label:10s} -> risk {d['risk_score']:>5} spoof {d['spoof_risk']:.3f} "
                  f"mism {d['speaker_mismatch']:.3f} {d['flags']} [{verdict}]")
            results.setdefault(label.split("@")[0], []).append(d)

    print()
    for tag, rows in results.items():
        avg = sum(r["risk_score"] for r in rows) / len(rows)
        alerts = sum(r["alert"] for r in rows)
        print(f"{tag}: avg risk {avg:.1f}, alerts {alerts}/{len(rows)}")


asyncio.run(main())
