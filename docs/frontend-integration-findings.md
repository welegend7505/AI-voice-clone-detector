# Frontend Integration Test — Findings (2026-09-06)

First live test of Member 1's frontend (AI-voice-clone-detector) against this
backend, with a real browser microphone. Verdict up front:

- **Contract: 100% correct.** Endpoints, query param, multipart field, hex WAV
  text frames, reconnect-to-same-call_id, null-tolerant frame parsing — all
  exactly per README §2. WAV encoder byte-perfect (48,000 samples + 44-byte
  header). No MediaRecorder/WebM anywhere, as documented.
- **One real bug found, root-caused, fix identified (frontend side):**
  real speech through the browser capture chain scores spoof 0.82–1.0
  (should be ~0.1–0.3) → constant false ALERTs on the trusted voice.
- **Backend: zero changes needed.**

## Evidence

Live mic session (trusted voice, should NOT alert):

```
[CHUNK] 3.00s 16000Hz x1ch WAV rms=0.0483   <- byte-perfect chunks, every ~3s
[ALERT] risk 91.9 spoof 0.892 mism 1.0
[ALERT] risk 84.0 spoof 1.0   mism 0.359
[ALERT] risk 96.3 spoof 1.0   mism 0.852    <- reproduced on a second call
```

Headless isolation experiment (this frontend's actual TS code, run in Node,
scored by this backend — `real_clean.wav`, which scores spoof 0.12 as a file):

```
A  clean 16k file through their encoder      spoof 0.151  encoder innocent
B  their code round-trip (integer ratio)     spoof 0.151  framing innocent
C  THEIR naive 48k->16k decim + hf noise     spoof 0.538  ALIASING
D  SAME noisy 48k sent raw, backend librosa  spoof 0.277  proper resample
```

C vs D is the same audio; only the resampler differs. Their `resampleLinear`
drops 2 of 3 samples with no anti-alias filter — real mic capture always has
energy above 8 kHz (fan, hiss, sibilance) which folds down into the speech
band, and both anti-spoof models read the folded artifacts as synthesis.
Browser DSP (`noiseSuppression`/`autoGainControl`/`echoCancellation`) is a
suspected additional contributor on top.

## Fixes for Member 1 (both in `src/lib/audio/microphoneCapture.ts`)

1. **Stop resampling client-side — send native-rate WAV.** In the chunker
   callback, replace `encodeWav16BitPcm(resampleLinear(window, ...), 16000)`
   with `encodeWav16BitPcm(window, context.sampleRate)`. The backend's
   librosa resampler handles any input rate correctly (proven by variant D:
   a 48 kHz chunk scored spoof 0.277 vs 0.538 through the client decimator).
   Cost: chunk grows ~3x (288 KB WAV → ~576 KB hex per 3 s) — negligible on
   localhost/demo wifi.
2. **Disable browser audio DSP** in `getUserMedia`:
   `echoCancellation: false, noiseSuppression: false, autoGainControl: false`.
   Nothing plays during capture, so echo cancellation is dead weight, and
   noise suppression carves vocoder-like holes in the spectrum.

Minor (optional): the enroll file input accepts any audio; add
`accept=".wav,audio/wav"` + a "convert phone recordings to WAV" hint — the
backend is WAV/PCM/FLAC/OGG only (rejects m4a/mp3/wma/amr with a named
reason in the `[ENROLL-REJ ...]` server log line).

## Verification after the fix

1. Backend-side headless check (no browser): re-run the experiment script
   from the frontend clone — variant C must drop toward D (~0.28).
2. Live: same mic test on `/live`. Expect spoof 0.1–0.3, risk < 35, no alert.
3. Re-check `speaker_mismatch`: with your own voice enrolled it should be
   < 0.3. If it stays > 0.5 after the capture fix, the enrolled WAV is not
   the same person as the mic — re-enroll.

## Clean-path control: the backend's detection is correct (same day)

After the browser tests, the same two audio sources scored through the
backend directly (file path, no browser) — `clone_test.py`:

```
user's real voice (3 x 3s):  risk 19.6-32.2, spoof 0.19-0.37, mism 0.18-0.23 -> 0/3 alerts
ElevenLabs "Roger" (3 x 3s): risk 50.1-67.5, spoof 0.39-0.58, mism 0.83-0.97 -> 3/3 alerts, both flags
```

Every browser-session score anomaly is therefore attributable to the capture
chain, not the models or fusion. Note: the ElevenLabs file is a STOCK voice
("Roger"), not a clone of the user — mismatch correctly fired high. A
genuine clone of the enrolled voice would keep mismatch low (~0.2), leaving
only spoof to carry the alert (risk ceiling 75 at current weights); spoof on
this ElevenLabs generation was only ~0.5, so a genuine clone of similar
quality could land near ~40-45 and slip under the 50 threshold. That is the
known P0 limitation (CLAUDE.md): generate a genuine clone and re-measure
with `clone_test.py` (swap the path) BEFORE the demo; tune weights/threshold
on that evidence if needed.

## Reproduction material (stays local, not committed)

- `AI-voice-clone-detector/experiment_resample.mjs` — the A/B/C/D experiment
- `AI-voice-clone-detector/pipeline_harness.mjs` — headless full-pipeline test
- Server console tags now log every chunk's true format: `[CHUNK ...]`,
  `[ENROLL ...]`, `[CHUNK-REJ ...]` (names the container, e.g. webm/m4a),
  `[CHUNK-ERR ...]`, `[ALERT ...]` — see README §2.
