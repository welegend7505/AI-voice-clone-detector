"""
RUN THIS FIRST -- before writing any more pipeline code.

Verifies both ML models download and load correctly on your actual
dev machine, standalone, with no FastAPI/WebSocket complexity in the way.
If either model fails here, you want to know at hour 0, not hour 10.

Usage:
    python test_models.py <path_to_real_voice_sample.wav> <path_to_second_sample.wav>

Try it with:
  1. Two recordings of the SAME real person -> expect high similarity_score,
     same_speaker=True, and a bonafide_score indicating genuine speech.
  2. A real recording + an AI-cloned recording of that same person (generate
     one with any free TTS/voice-clone tool for testing) -> expect the
     anti-spoof score to shift and ideally a lower bonafide_score.

Use the results from a handful of these tests to pick real thresholds
via anti_spoof.calibrate_threshold() instead of trusting the placeholder
math in api/routes.py.
"""

import sys

from models import anti_spoof, speaker


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_models.py <wav1> <wav2>")
        sys.exit(1)

    wav1, wav2 = sys.argv[1], sys.argv[2]

    print("Loading + testing AASIST anti-spoof model...")
    result = anti_spoof.score_audio(wav1)
    print("  bonafide_score (wav1):", result["bonafide_score"])

    print("\nLoading + testing SpeechBrain speaker verification...")
    result = speaker.verify_speaker(wav1, wav2)
    print("  similarity_score:", result["similarity_score"])
    print("  same_speaker:", result["same_speaker"])

    print("\nBoth models loaded and ran successfully. Safe to proceed to the API layer.")


if __name__ == "__main__":
    main()
