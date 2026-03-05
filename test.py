import os
import tempfile

import numpy as np
import soundfile as sf

from config import Config
from main import DoggyNeuralGlitch


def create_test_audio(path: str, duration: float = 3.0, sr: int = 22050):
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)

    vibrato = np.sin(2 * np.pi * 5 * t) * 0.5
    a_note = np.sin(2 * np.pi * 440 * t + vibrato) * 0.4   # vibrato A (440 Hz)
    high = np.sin(2 * np.pi * 880 * t + np.sin(2 * np.pi * 6 * t) * 0.3) * 0.2
    bass = np.sin(2 * np.pi * 220 * t) * 0.3
    noise = np.random.randn(len(t)) * 0.05

    audio = a_note + high + bass + noise
    audio /= np.abs(audio).max()
    sf.write(path, audio.astype(np.float32), sr)
    print(f"Created test audio: {path} ({duration}s at {sr}Hz)")


if __name__ == '__main__':
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        test_audio = f.name

    try:
        create_test_audio(test_audio)

        glitcher = DoggyNeuralGlitch(Config())
        glitcher.process(
            audio_path=test_audio,
            output_dir='./test_output',
            mode='mixed',
            max_frames=5,
        )

        frames = sorted(os.listdir('./test_output'))
        print(f"\nGenerated {len(frames)} frames:")
        for name in frames:
            path = os.path.join('./test_output', name)
            print(f"  {name}: {os.path.getsize(path):,} bytes")

        assert len(frames) == 5, f"Expected 5 frames, got {len(frames)}"
        print("\nAll checks passed!")
    finally:
        os.unlink(test_audio)
