import numpy as np

from config import Config


class GlitchEffects:
    @staticmethod
    def bitcrush(img: np.ndarray, intensity: float) -> np.ndarray:
        """Reduce bit depth based on audio intensity."""
        bit_depth = max(1, int(8 - intensity * 7))
        levels = 2 ** bit_depth
        img_float = img.astype(np.float32)
        quantized = np.floor(img_float / 256.0 * levels) * (256.0 / levels)
        return np.clip(quantized, 0, 255).astype(np.uint8)

    @staticmethod
    def rgb_split(img: np.ndarray, intensity: float,
                  spectral_centroid: float) -> np.ndarray:
        """Shift red/blue channels horizontally or vertically."""
        offset = int(intensity * 50)
        if offset == 0:
            return img
        result = img.copy()
        if spectral_centroid > 0.5:  # horizontal
            result[:, offset:, 0] = img[:, :-offset, 0]
            result[:, :-offset, 2] = img[:, offset:, 2]
        else:  # vertical
            result[offset:, :, 0] = img[:-offset, :, 0]
            result[:-offset, :, 2] = img[offset:, :, 2]
        return result

    @staticmethod
    def datamosh(img: np.ndarray, intensity: float,
                 audio_bytes: bytes) -> np.ndarray:
        """Inject raw audio bytes directly into image data."""
        n_corrupt = int(img.size * intensity * 0.3)
        if n_corrupt == 0 or not audio_bytes:
            return img
        flat = img.flatten().copy()
        indices = np.random.choice(len(flat), n_corrupt, replace=False)
        raw = np.frombuffer(audio_bytes, dtype=np.uint8)
        if len(raw) < n_corrupt:
            raw = np.tile(raw, n_corrupt // len(raw) + 1)
        flat[indices] = raw[:n_corrupt]
        return flat.reshape(img.shape)

    @staticmethod
    def chromatic_aberration(img: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate lens distortion by progressively offsetting RGB channels."""
        offset = max(1, int(intensity * 20))
        offset2 = min(offset * 2, img.shape[1] - 1)
        result = img.copy()
        # Red: no shift; Green: +offset; Blue: +offset*2
        result[:, offset:, 1] = img[:, :-offset, 1]
        result[:, offset2:, 2] = img[:, :-offset2, 2]
        return result

    @staticmethod
    def pixel_sort(img: np.ndarray, intensity: float) -> np.ndarray:
        """Sort pixels by brightness within randomly selected rows."""
        n_rows = int(intensity * img.shape[0])
        if n_rows == 0:
            return img
        result = img.copy()
        rows = np.random.choice(img.shape[0], n_rows, replace=False)
        threshold = 128
        for row in rows:
            brightness = result[row].mean(axis=1)
            mask = brightness > threshold
            if mask.any():
                idx = np.where(mask)[0]
                pixels = result[row, idx]
                pixels = pixels[np.argsort(pixels.mean(axis=1))]
                result[row, idx] = pixels
        return result

    @staticmethod
    def block_glitch(img: np.ndarray, bass_intensity: float) -> np.ndarray:
        """Randomly displace rectangular blocks driven by bass intensity."""
        h, w = img.shape[:2]
        n_blocks = max(1, int(bass_intensity * 10))
        result = img.copy()
        for _ in range(n_blocks):
            bh = np.random.randint(32, 65)
            bw = np.random.randint(32, 65)
            y = np.random.randint(0, max(1, h - bh))
            x = np.random.randint(0, max(1, w - bw))
            dy = np.random.randint(-20, 21)
            dx = np.random.randint(-20, 21)
            ny = int(np.clip(y + dy, 0, h - bh))
            nx = int(np.clip(x + dx, 0, w - bw))
            result[ny:ny + bh, nx:nx + bw] = img[y:y + bh, x:x + bw]
        return result

    @staticmethod
    def scanlines(img: np.ndarray, line_height: int = 4,
                  intensity: float = 0.3) -> np.ndarray:
        """Add CRT monitor scanline effect."""
        result = img.astype(np.float32)
        for y in range(0, img.shape[0], line_height * 2):
            result[y:y + line_height] *= (1.0 - intensity)
        return np.clip(result, 0, 255).astype(np.uint8)


class AudioDrivenGlitcher:
    def __init__(self, config: Config):
        self.config = config

    def apply_audio_driven_glitches(self, img: np.ndarray, features: dict,
                                     mode: str) -> np.ndarray:
        intensity = features.get('intensity', 0.5)
        bass = features.get('bass', 0.5)
        mid = features.get('mid', 0.5)
        treble = features.get('treble', 0.5)
        centroid = features.get('spectral_centroid', 0.5)
        audio_bytes = features.get('raw_bytes', b'')

        if mode == 'bitcrush':
            return GlitchEffects.bitcrush(img, intensity)
        if mode == 'rgb_split':
            return GlitchEffects.rgb_split(img, intensity, centroid)
        if mode == 'datamosh':
            return GlitchEffects.datamosh(img, intensity, audio_bytes)
        if mode == 'mixed':
            return self._apply_mixed(img, intensity, bass, mid, treble,
                                     centroid, audio_bytes)
        return img

    def _apply_mixed(self, img, intensity, bass, mid, treble, centroid,
                     audio_bytes):
        # Structural changes first
        if bass > 0.6:
            img = GlitchEffects.block_glitch(img, bass)
        # Detail changes
        if treble > 0.5:
            img = GlitchEffects.rgb_split(img, intensity, centroid)
        # Tonal changes
        if mid > 0.4:
            img = GlitchEffects.bitcrush(img, mid)
        # Always apply chromatic aberration
        img = GlitchEffects.chromatic_aberration(img, intensity)
        # Probabilistic pixel sort
        if np.random.random() < intensity:
            img = GlitchEffects.pixel_sort(img, intensity)
        return img
