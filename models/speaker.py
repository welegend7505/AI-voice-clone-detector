"""
Speaker verification using SpeechBrain's pretrained ECAPA-TDNN model.

Model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
Trained on VoxCeleb1+2. Compares two audio files and returns a
similarity score + same/different speaker prediction.

Requires: pip install speechbrain
First run downloads the checkpoint via huggingface_hub -- needs internet
access to huggingface.co.

WINDOWS NOTE (hit live on 2026-09-05): we deliberately do NOT use
`verify_files(path, path)`. SpeechBrain's path handling splits on forward
slashes only, so an absolute Windows path like C:\\Users\\...\\x.wav gets
mangled into "<cwd>\\C:\\Users\\...\\x.wav" and loading fails. Instead we
load the audio ourselves (soundfile, 16 kHz mono -- exactly what
streaming/audio_utils.py already produces) and pass tensors to
`verify_batch`, which is the same public API without the path layer.
"""

import librosa
import numpy as np
import soundfile as sf
import torch

_verifier = None


def get_verifier():
    global _verifier
    if _verifier is None:
        # Import path varies slightly by speechbrain version.
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:
            from speechbrain.pretrained import SpeakerRecognition  # older versions

        # On Windows without Developer Mode, speechbrain's default SYMLINK
        # strategy fails with "required privilege is not held" (WinError 1314).
        # COPY is slower on first download but works everywhere.
        from speechbrain.utils.fetching import LocalStrategy

        _verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            local_strategy=LocalStrategy.COPY,
        )
    return _verifier


def _load_as_16k_tensor(wav_path: str) -> torch.Tensor:
    """Any wav -> float32 mono 16 kHz torch tensor of shape (1, time)."""
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # collapse stereo to mono
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    data = np.ascontiguousarray(data, dtype=np.float32)
    return torch.from_numpy(data).unsqueeze(0)  # (1, time)


def verify_speaker(enrolled_wav_path: str, live_wav_path: str) -> dict:
    """
    Args:
        enrolled_wav_path: path to the trusted person's enrolled reference sample
        live_wav_path: path to the current live audio chunk
    Returns:
        dict with similarity_score (float) and same_speaker (bool)
    """
    verifier = get_verifier()
    enrolled = _load_as_16k_tensor(enrolled_wav_path)
    live = _load_as_16k_tensor(live_wav_path)
    score, prediction = verifier.verify_batch(enrolled, live)  # threshold 0.25
    return {
        "similarity_score": float(score),
        "same_speaker": bool(prediction),
    }
