"""Build the stress-test sample set in samples/stress/ (run once, 2026-09-06).

Starts from:
  - samples/real/OSR_us_000_0010_8k.wav  -> enrollment reference (real_enroll.wav)
  - samples/real/OSR_us_000_0011_8k.wav  -> clean live take of same speaker
  - samples/stress/src/*.mp3             -> edge-tts neural TTS clips (the
    modern synthesis stand-in for a voice clone; see PROVENANCE-stress.md)

Everything is written as 16 kHz mono PCM16 wav (what the browser demo sends).
Degradations are applied AFTER resampling to 16 kHz so they land in the same
domain the live pipeline sees. RNG is seeded -> fully reproducible.

Conditions per the stress-test brief:
  white15/white5   additive broadband noise at 15 / 5 dB SNR
  pink10           pink (1/f) noise at 10 dB SNR -- steady HVAC/fan-like
  babble10         background speech (another TTS voice) at 10 dB SNR
  revsmall/revlarge synthetic room reverb, RT60 ~0.35 s / ~0.8 s
  phonespk         phone-on-speaker sim: 300-3400 Hz bandpass + 15 dB white
                   noise + soft saturation
  far              speaker far from mic: -20 dB gain + small-room reverb
  gainNN           pure level sweep (no added noise) -- how quiet is too quiet

Usage:  python make_stress_samples.py
"""

import glob
import os

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
OUT = os.path.join("samples", "stress")
RNG = np.random.default_rng(26104)  # = SIH26104, deterministic set


def load_16k(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return np.asarray(data, dtype=np.float32)


def save(name, data):
    peak = np.abs(data).max()
    if peak > 1.0:  # keep degradations from clipping the int16 write
        data = data / peak * 0.99
    sf.write(os.path.join(OUT, name), data, SR, subtype="PCM_16")


def add_noise(speech, snr_db, kind="white"):
    """Mix noise at a target SNR relative to speech RMS power."""
    n = len(speech)
    if kind == "white":
        noise = RNG.standard_normal(n).astype(np.float32)
    elif kind == "pink":
        # FFT-shape white noise so power ~ 1/f
        spec = np.fft.rfft(RNG.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1.0 / SR)
        freqs[0] = freqs[1]  # avoid div-by-zero at DC
        noise = np.fft.irfft(spec / np.sqrt(freqs), n).astype(np.float32)
    else:
        raise ValueError(kind)
    p_s = float((speech ** 2).mean())
    p_n = float((noise ** 2).mean())
    noise *= np.sqrt(p_s / p_n / (10 ** (snr_db / 10)))
    return speech + noise


def add_babble(speech, babble_path, snr_db):
    """Background talking (a different voice) at a target SNR."""
    bab = load_16k(babble_path)
    if len(bab) < len(speech):  # loop to cover
        bab = np.tile(bab, int(np.ceil(len(speech) / len(bab))))
    start = RNG.integers(0, len(bab) - len(speech) + 1)
    bab = bab[start:start + len(speech)].astype(np.float32)
    p_s = float((speech ** 2).mean())
    p_b = float((bab ** 2).mean())
    bab *= np.sqrt(p_s / p_b / (10 ** (snr_db / 10)))
    return speech + bab


def reverb(speech, t60_s):
    """Convolve with a synthetic exponentially-decaying-noise impulse response."""
    ir_len = int(t60_s * SR)
    t = np.arange(ir_len) / SR
    ir = RNG.standard_normal(ir_len).astype(np.float32) * np.exp(-6.91 * t / t60_s)
    ir[0] = 0.0
    ir /= np.sqrt((ir ** 2).sum())          # unit energy -> speech level kept
    wet = signal.fftconvolve(speech, ir)[: len(speech) + ir_len - 1][: len(speech)]
    return 0.7 * speech + 0.5 * wet         # mild direct/wet mix


def phone_speaker(speech):
    """Call played on a phone speaker, recorded across a table."""
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    bandlimited = signal.sosfilt(sos, speech).astype(np.float32)
    noisy = add_noise(bandlimited, 15.0, "white")
    return np.tanh(2.0 * noisy) * 0.5       # soft saturation like a tiny driver


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "src"), exist_ok=True)

    # --- clean real speech (16 kHz) ---
    enroll = load_16k(os.path.join("samples", "real", "OSR_us_000_0010_8k.wav"))
    clean = load_16k(os.path.join("samples", "real", "OSR_us_000_0011_8k.wav"))
    save("real_enroll.wav", enroll)
    save("real_clean.wav", clean)

    # --- noisy real speech (same take, one degradation each) ---
    save("real_white15.wav", add_noise(clean, 15, "white"))
    save("real_white5.wav", add_noise(clean, 5, "white"))
    save("real_pink10.wav", add_noise(clean, 10, "pink"))
    save("real_babble10.wav", add_babble(clean, os.path.join(OUT, "src", "tts_US-Jenny_1.mp3"), 10))
    save("real_revsmall.wav", reverb(clean, 0.35))
    save("real_revlarge.wav", reverb(clean, 0.8))
    save("real_phonespk.wav", phone_speaker(clean))
    save("real_far.wav", add_noise(reverb(clean * 0.1, 0.35), 25, "pink"))

    # --- pure level sweep on the clean take ---
    for pct in (50, 25, 10, 5, 2):
        save(f"real_gain{pct}.wav", clean * (pct / 100))

    # --- neural TTS clips: clean + same degradation suite (subset) ---
    mp3s = sorted(glob.glob(os.path.join(OUT, "src", "*.mp3")))
    if not mp3s:
        raise SystemExit("samples/stress/src/*.mp3 missing -- generate with edge-tts first")
    for mp3 in mp3s:
        tag = os.path.splitext(os.path.basename(mp3))[0].replace("tts_", "")  # e.g. US-Guy_1
        data = load_16k(mp3)
        save(f"tts_{tag}_clean.wav", data)
    guy = load_16k(os.path.join(OUT, "src", "tts_US-Guy_1.mp3"))
    save("tts_US-Guy_1_white15.wav", add_noise(guy, 15, "white"))
    save("tts_US-Guy_1_white5.wav", add_noise(guy, 5, "white"))
    save("tts_US-Guy_1_pink10.wav", add_noise(guy, 10, "pink"))
    save("tts_US-Guy_1_phonespk.wav", phone_speaker(guy))
    save("tts_US-Guy_1_revlarge.wav", reverb(guy, 0.8))
    save("tts_US-Guy_1_babble10.wav", add_babble(guy, os.path.join(OUT, "src", "tts_US-Jenny_1.mp3"), 10))

    # --- appended 2026-09-06: locate the babble break-point (keep AFTER all
    # draws above so earlier files stay byte-identical under the seeded RNG) ---
    save("real_babble15.wav", add_babble(clean, os.path.join(OUT, "src", "tts_US-Jenny_1.mp3"), 15))
    save("real_babble20.wav", add_babble(clean, os.path.join(OUT, "src", "tts_US-Jenny_1.mp3"), 20))

    made = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*.wav")))
    print(f"wrote {len(made)} wavs to {OUT}:")
    for m in made:
        print(" ", m)


if __name__ == "__main__":
    main()
