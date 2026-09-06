"""
Helpers for turning incoming audio chunks (from the frontend WebSocket
stream) into 16kHz mono wav files the models can consume.

Chunk size / encoding format MUST be agreed with Member 1 (frontend) --
suggested default: ~2.5-3 sec windows, 16kHz mono PCM/WAV, hex or
base64-encoded over the WebSocket JSON message.
"""

import io
import tempfile

import librosa
import soundfile as sf

TARGET_SR = 16000


def bytes_to_wav_file(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Writes raw audio bytes to a temp wav file, resampled to 16kHz mono.
    Returns the temp file path (caller is responsible for cleanup if
    running long-lived processes -- fine to leave for a hackathon demo).

    Raises soundfile.LibsndfileError (or ValueError) if the bytes are not
    audio soundfile can parse -- callers should catch and return a clean
    4xx/error response instead of letting it bubble up as a 500.
    """
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse stereo to mono
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    sf.write(tmp.name, data, TARGET_SR)
    tmp.close()  # Windows won't let callers delete the file while we hold it open
    return tmp.name


def chunk_stats(wav_path: str) -> dict:
    """
    Duration and RMS level of a wav file, for sanity-checking live chunks.
    Returns {"duration_s": float, "rms": float} -- rms near 0 means silence.
    """
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / sr
    rms = float((data ** 2).mean() ** 0.5) if len(data) else 0.0
    return {"duration_s": duration, "rms": rms}


def probe_audio(audio_bytes: bytes) -> dict:
    """
    What actually arrived, BEFORE 16kHz normalization: container, sample
    rate, channels, duration. Feeds the per-chunk intake log so a frontend
    format mismatch (e.g. MediaRecorder's webm/opus instead of WAV) is
    visible on the server console the moment it happens.

    Returns {"format", "sr", "channels", "duration_s"}; the last three are
    None when the bytes are not parseable audio at all -- "format" then
    carries a best-effort container hint from the magic bytes.
    """
    try:
        meta = sf.info(io.BytesIO(audio_bytes))
        return {"format": meta.format, "sr": meta.samplerate,
                "channels": meta.channels, "duration_s": meta.duration}
    except Exception:
        return {"format": _container_hint(audio_bytes), "sr": None,
                "channels": None, "duration_s": None}


def _container_hint(audio_bytes: bytes) -> str:
    """Best-effort container guess from magic bytes (unreadable input)."""
    if audio_bytes[:4] == b"RIFF":
        return "wav? (RIFF header but body unparseable -- truncated?)"
    if audio_bytes[:4] == b"\x1aE\xdf\xa3":
        return "webm/matroska (MediaRecorder default) -- convert to WAV client-side"
    if audio_bytes[:4] == b"OggS":
        return "ogg/opus"
    if audio_bytes[:3] == b"ID3":
        return "mp3"
    if audio_bytes[4:8] == b"ftyp":
        return "m4a/mp4 (phone recording) -- convert to WAV client-side"
    if audio_bytes[:4] == b"fLaC":
        return "flac (should have parsed?)"
    if audio_bytes[:4] == b"\x30\x26\xb2\x75":
        return "wma/asf"
    if audio_bytes[:5] == b"#!AMR":
        return "amr"
    return "unknown binary"
