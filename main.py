import argparse
import os

import librosa
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from config import BANNER, Config
from glitch_effects import AudioDrivenGlitcher
from neural_core import TinyVAE


class DoggyNeuralGlitch:
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(42069)
        np.random.seed(42069)
        self.model = TinyVAE(
            input_dim=128,
            latent_dim=config.LATENT_DIM,
            channels=config.DECODER_CHANNELS,
        )
        self.model.eval()
        self.glitcher = AudioDrivenGlitcher(config)

    def extract_audio_features(self, audio_path: str,
                                chunk_duration: float = None) -> list[dict]:
        if chunk_duration is None:
            chunk_duration = self.config.CHUNK_DURATION
        y, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        for start in range(0, len(y) - chunk_samples + 1, chunk_samples):
            chunk = y[start:start + chunk_samples]
            chunks.append(self._extract_chunk_features(chunk, sr))
        return chunks

    def _extract_chunk_features(self, chunk: np.ndarray, sr: int) -> dict:
        # MFCCs -> padded/truncated to 128 dims, normalized
        mfccs = librosa.feature.mfcc(
            y=chunk, sr=sr, n_mfcc=self.config.N_MFCC,
            hop_length=self.config.HOP_LENGTH,
        )
        mfcc_flat = mfccs.flatten()
        if len(mfcc_flat) >= 128:
            mfcc_vec = mfcc_flat[:128]
        else:
            mfcc_vec = np.pad(mfcc_flat, (0, 128 - len(mfcc_flat)))
        std = mfcc_vec.std()
        if std > 0:
            mfcc_vec = (mfcc_vec - mfcc_vec.mean()) / std

        # RMS energy -> intensity [0, 1]
        rms = float(librosa.feature.rms(y=chunk).mean())
        intensity = float(np.clip(rms / 0.1, 0.0, 1.0))

        # Spectral centroid -> normalized [0, 1]
        centroid = float(librosa.feature.spectral_centroid(y=chunk, sr=sr).mean())
        centroid_norm = float(np.clip(centroid / (sr / 2), 0.0, 1.0))

        # Frequency band energies (relative, normalized to [0, 1])
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
        mean_energy = fft.mean() + 1e-8
        bass = float(fft[(freqs >= 20) & (freqs < 250)].mean() / mean_energy)
        mid = float(fft[(freqs >= 250) & (freqs < 4000)].mean() / mean_energy)
        treble = float(fft[freqs >= 4000].mean() / mean_energy)
        band_max = max(bass, mid, treble, 1e-8)
        bass = float(np.clip(bass / band_max, 0.0, 1.0))
        mid = float(np.clip(mid / band_max, 0.0, 1.0))
        treble = float(np.clip(treble / band_max, 0.0, 1.0))

        # Raw int16 bytes for datamoshing
        raw_bytes = (chunk * 32767).astype(np.int16).tobytes()

        return {
            'mfcc_vec': mfcc_vec.astype(np.float32),
            'intensity': intensity,
            'spectral_centroid': centroid_norm,
            'bass': bass,
            'mid': mid,
            'treble': treble,
            'raw_bytes': raw_bytes,
        }

    def _generate_plasma(self, features: dict) -> np.ndarray:
        """Audio-driven plasma: sum of sine waves keyed to MFCC/band values."""
        h, w = self.config.IMAGE_SIZE
        mfcc = features['mfcc_vec']
        bass, mid, treble = features['bass'], features['mid'], features['treble']
        intensity = features['intensity']

        x = np.linspace(0, 2 * np.pi, w, dtype=np.float32)
        y = np.linspace(0, 2 * np.pi, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Bass → low-freq red structure; mid → green rhythm; treble → blue detail
        r = (np.sin(xx * (bass * 4 + 1) + mfcc[0] * np.pi) +
             np.sin(yy * (mid  * 2 + 0.5) + mfcc[1] * np.pi)) * 0.5
        g = (np.sin(yy * (mid  * 4 + 1) + mfcc[2] * np.pi) +
             np.cos(xx * (treble * 2 + 0.5) + mfcc[3] * np.pi)) * 0.5
        b = (np.cos((xx + yy) * (treble * 3 + 1) + mfcc[4] * np.pi) +
             np.sin(xx * yy * intensity * 2 + mfcc[5] * np.pi)) * 0.5

        img = np.stack([r, g, b], axis=2)
        return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def _vae_texture(self, features: dict) -> np.ndarray:
        """VAE decode: used as an additive texture layer over the plasma base."""
        x = torch.tensor(features['mfcc_vec']).unsqueeze(0)
        with torch.no_grad():
            z, _, _ = self.model.encode(x)
            img_tensor = self.model.decode(z)
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        return ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def _generate_drakonix(self, features: dict) -> np.ndarray:
        """Fursona palette stripes + audio-driven Perlin-style noise — ported from favicon.rs."""
        h, w = self.config.IMAGE_SIZE
        palette = self.config.COLOR_PALETTES['drakonix']
        mfcc = features['mfcc_vec']
        bass, mid, treble = features['bass'], features['mid'], features['treble']
        intensity = features['intensity']

        # Audio-driven parameters mirroring the favicon generator's random choices:
        #   spectral centroid → stripe direction  (favicon: rng.gen_bool(0.5))
        #   bass              → stripe count 3–6  (favicon: rng.gen_range(3..=6))
        #   treble            → noise scale 3–9   (favicon: rng.gen_range(3.0..9.0))
        #   intensity         → noise strength 25–70 (favicon: rng.gen_range(25.0..70.0))
        horizontal = features['spectral_centroid'] > 0.5
        num_stripes = int(bass * 3 + 3)  # 3–6
        noise_scale = treble * 6 + 3     # 3–9
        noise_strength = intensity * 45 + 25  # 25–70

        # Assign one palette color per stripe, keyed by MFCC so it's audio-deterministic
        stripe_colors = [
            palette[int(abs(mfcc[i % len(mfcc)]) * len(palette)) % len(palette)]
            for i in range(num_stripes)
        ]

        img = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                axis = y if horizontal else x
                size = h if horizontal else w
                idx = min(int((axis / size) * num_stripes), num_stripes - 1)
                img[y, x] = stripe_colors[idx]

        # Perlin-like noise: gaussian-smoothed random field, same technique as favicon.rs
        raw_noise = np.random.randn(h, w).astype(np.float32)
        smooth = gaussian_filter(raw_noise, sigma=max(noise_scale, 0.1))
        smooth = smooth / (smooth.std() + 1e-8)  # normalise to ~[-1, 1]
        shift = (smooth * noise_strength)[..., np.newaxis]  # (h, w, 1)

        img = np.clip(img + shift, 0, 255).astype(np.uint8)
        return img

    def generate_frame(self, features: dict, mode: str) -> np.ndarray:
        plasma = self._generate_plasma(features)

        if mode == 'chaos':
            # Pure plasma seeded by audio intensity — no VAE
            return plasma

        # Neural mode: blend plasma (color/structure) with VAE (texture)
        vae = self._vae_texture(features)
        blended = (plasma.astype(np.float32) * 0.7 + vae.astype(np.float32) * 0.3)
        return blended.clip(0, 255).astype(np.uint8)

    def process(self, audio_path: str, output_dir: str, mode: str = 'mixed',
                max_frames: int = None, chunk_duration: float = None):
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading audio: {audio_path}")
        chunks = self.extract_audio_features(audio_path, chunk_duration)
        if max_frames is not None:
            chunks = chunks[:max_frames]

        print(f"Generating {len(chunks)} frames in '{mode}' mode...")
        for i, features in enumerate(tqdm(chunks)):
            img = self._generate_and_glitch(features, mode)
            Image.fromarray(img).save(os.path.join(output_dir, f"frame_{i:04d}.png"))

        print(f"Done! {len(chunks)} frames saved to {output_dir}")

    def _generate_and_glitch(self, features: dict, mode: str) -> np.ndarray:
        if mode == 'chaos':
            return self.generate_frame(features, 'chaos')
        if mode == 'drakonix':
            img = self._generate_drakonix(features)
            return self.glitcher.apply_audio_driven_glitches(img, features, 'mixed')
        # All other modes: neural base + effect pass
        img = self.generate_frame(features, 'neural')
        effect_mode = mode if mode != 'neural' else 'mixed'
        return self.glitcher.apply_audio_driven_glitches(img, features, effect_mode)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description='DoggyNeuralGlitch - Audio-to-Glitch-Art Generator'
    )
    parser.add_argument('--input', required=True,
                        help='Input audio file (MP3/WAV)')
    parser.add_argument('--output', default='./output',
                        help='Output directory for PNG frames')
    parser.add_argument('--mode', default='mixed',
                        choices=['neural', 'chaos', 'mixed',
                                 'bitcrush', 'rgb_split', 'datamosh', 'drakonix'],
                        help='Generation mode (default: mixed)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to generate')
    parser.add_argument('--chunk-duration', type=float, default=None,
                        help='Audio chunk duration in seconds (default: 0.5)')
    args = parser.parse_args()

    glitcher = DoggyNeuralGlitch(Config())
    glitcher.process(
        audio_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        max_frames=args.max_frames,
        chunk_duration=args.chunk_duration,
    )


if __name__ == '__main__':
    main()
