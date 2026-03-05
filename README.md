# 🐕 DoggyNeuralGlitch - Complete Project Specification

## Project Genesis

This project was created in response to a social media post speculating about a future where extreme model quantization (1-bit) enables demoscene-style neural art projects. The vision: an LLM that ingests raw MP3 data and outputs glitch art in real-time, using only 100MB RAM and running CPU-only.

While true 1-bit LLMs processing raw audio don't exist yet, this project demonstrates the concept using existing technologies mashed together creatively.

---

## Core Concept

**DoggyNeuralGlitch** is a demoscene-inspired audio-to-visual glitch art generator that:
- Takes MP3/WAV audio files as input
- Processes them through a tiny neural network (~400k parameters, 2-4MB)
- Applies audio-driven corruption effects
- Outputs unique glitch art frames (one per audio chunk)
- Runs entirely on CPU with ~50-80MB RAM
- Generates at 30 FPS (0.033s chunks) and exports directly to MP4 via ffmpeg

The name "Doggy" reflects the philosophy: neural networks are good boys who sometimes make beautiful mistakes.

---

## Technical Architecture

### Pipeline Overview
```
Audio File → Audio Processing → Base Image (Plasma / VAE blend) → Glitch Effects → PNG Frames → MP4
```

### Component 1: Audio Processing (librosa)

**File**: `main.py` - `extract_audio_features()` method

**Process**:
1. Load audio at 22050 Hz sample rate
2. Split into chunks (default: 0.5 seconds each)
3. Extract features per chunk:
   - **MFCCs**: 13 Mel-frequency cepstral coefficients (timbral features)
   - **RMS Energy**: Overall loudness, normalized to 0-1 (called "intensity")
   - **Spectral Centroid**: Brightness measure (center of mass of spectrum)
   - **Frequency Bands**: Bass (20-250Hz), Mid (250-4000Hz), Treble (4000+Hz)
   - **Raw Bytes**: int16 audio data for datamoshing

**Key Parameters**:
- Sample rate: 22050 Hz
- Hop length: 512 samples
- N_MFCC: 13 coefficients
- Chunk duration: 0.0333 seconds (default = 1/30fps, configurable)

### Component 2: Neural Core (PyTorch)

**File**: `neural_core.py`

**Architecture**: Mini VAE (Variational Autoencoder)

**Encoder**:
- Input: 128-dim audio feature vector (flattened/padded MFCCs)
- Layers: Linear(128→128)→ReLU→Linear(128→64)→ReLU→Linear(64→128)
- Output: 128-dim vector split into mean (64-dim) and logvar (64-dim)

**Decoder**:
- Input: 64-dim latent vector (from reparameterization)
- Linear(64→4096)→ReLU→Reshape(16, 16, 16) (spatial base)
- ConvTranspose2d(16→64, 4×4, stride=2)→ReLU → (64, 32, 32)
- ConvTranspose2d(64→32, 4×4, stride=2)→ReLU → (32, 64, 64)
- ConvTranspose2d(32→16, 4×4, stride=2)→ReLU → (16, 128, 128)
- ConvTranspose2d(16→8, 4×4, stride=2)→ReLU → (8, 256, 256)
- ConvTranspose2d(8→3, 4×4, stride=2)→Tanh → (3, 512, 512)
- Output: 512×512×3 image tensor in range [-1, 1]

**Reparameterization Trick**:
```python
z = mean + std * epsilon
where std = exp(0.5 * logvar)
      epsilon ~ N(0, 1)
```

**Chaos Mode**:
- Pure audio-driven plasma — VAE is skipped entirely
- Plasma frequencies/phases seeded directly from audio features

**Initialization**:
- Xavier normal initialization with gain=2.0 (saturates activations for visual interest)
- Fixed seed (42069) for reproducible results
- **NO TRAINING REQUIRED** - VAE acts as a texture layer over the plasma base

**Total Parameters**: ~400,000 (encoder ~33k + conv decoder ~360k)
**Model Size**: ~2-4 MB

### Component 3: Glitch Effects Pipeline

**File**: `glitch_effects.py`

**Available Effects**:

1. **Bitcrush**: Reduce bit depth from 8-bit to 1-8 bits based on audio intensity
   ```python
   levels = 2 ** bit_depth
   img_quantized = floor(img / 256 * levels) * (256 / levels)
   ```

2. **RGB Split**: Shift red/blue channels horizontally or vertically
   - Offset determined by audio intensity (0-50 pixels)
   - Direction determined by spectral centroid (high=horizontal, low=vertical)

3. **Datamosh**: Inject raw audio bytes directly into image data
   - Corruption strength = audio intensity * 0.3
   - Randomly replace pixel values with audio bytes

4. **Chromatic Aberration**: Simulate lens distortion by offsetting RGB channels progressively
   - Red: no offset
   - Green: offset pixels
   - Blue: offset * 2 pixels

5. **Pixel Sorting**: Sort pixels by brightness within segments
   - Threshold determines which pixels to sort
   - Number of rows affected = audio intensity * height

6. **Block Glitching**: Randomly displace rectangular blocks
   - Block size: 32-64 pixels
   - Number of blocks determined by bass intensity

7. **Scanlines**: Add CRT monitor effect (optional)
   - Line height: 4 pixels
   - Intensity: 0.3

**Audio-to-Effect Mapping (Mixed Mode)**:
```python
if bass_intensity > 0.3:
    apply block glitches (structural changes)

if treble_intensity > 0.25:
    apply RGB split (detail changes)

if mid_intensity > 0.2:
    apply bitcrush (tonal changes)

always:
    apply chromatic aberration (scaled by intensity)

if random() < intensity:
    apply pixel sorting
```

### Component 4: Configuration

**File**: `config.py`

**Key Settings**:
```python
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_MFCC = 13
CHUNK_DURATION = 0.5       # override via --chunk-duration; justfile default is 0.0333
IMAGE_SIZE = (512, 512)
LATENT_DIM = 64
DECODER_CHANNELS = [64, 32, 16, 8, 3]  # ConvTranspose2d channel progression
USE_GPU = False
```

**Color Palettes**:
- vaporwave: [(255,113,206), (103,58,183), (0,191,255)]
- cyberpunk: [(255,0,110), (0,255,255), (255,255,0)]
- grayscale: Evenly spaced grays
- demoscene: [(0,255,0), (255,0,255), (0,255,255)]
- drakonix: deep navy, dark navy, bright purple, lilac, cyan, light cyan, medium purple, pale lavender, off-white

**Effect Weights** (for mixed mode):
```python
EFFECT_WEIGHTS = {
    'neural': 0.3,
    'bitcrush': 0.2,
    'datamosh': 0.2,
    'rgb_split': 0.2,
    'chaos': 0.1
}
```

---

## File Structure

```
DoggyNeuralGlitch/
├── README.md              # This file - complete specification
├── LICENSE                # MIT license
├── pyproject.toml         # uv project + dependencies
├── requirements.txt       # Pip-compatible dependency list
├── justfile               # Dev task runner (just run, just test, etc.)
├── .gitignore
├── config.py             # Config class, color palettes, ASCII art banner
├── neural_core.py        # Tiny VAE (encoder + conv decoder)
├── glitch_effects.py     # 7 glitch effects + AudioDrivenGlitcher
├── main.py               # Pipeline: plasma generator, VAE blend, CLI
└── test.py               # Quick test with synthetic audio
```

---

## Dependencies

**Package manager**: [`uv`](https://github.com/astral-sh/uv) (recommended)

```bash
uv sync          # install all deps from pyproject.toml
uv run main.py --input song.mp3 --output ./output
```

**requirements.txt**:
```
numpy>=1.24.0
Pillow>=10.0.0
librosa>=0.10.0
soundfile>=0.12.0
torch>=2.0.0
tqdm>=4.65.0
scipy>=1.10.0
```

**Key Libraries**:
- `librosa`: Audio analysis and feature extraction
- `torch`: Neural network with conv decoder (CPU-only)
- `numpy`: Array operations for glitch effects
- `PIL`: Image I/O and basic manipulation
- `soundfile`: Audio file I/O

---

## Usage Patterns

### Basic Command Line Usage
```bash
# Process entire audio file (30fps default)
just run song.mp3

# Convenience targets — one per main mode
just run-mixed    song.mp3
just run-chaos    song.mp3
just run-neural   song.mp3
just run-drakonix song.mp3

# Preview first 90 frames (~3 seconds at 30fps)
just run-frames song.mp3 90

# Override fps / chunk duration together (must stay in sync: chunk-duration = 1/fps)
just run song.mp3 ./output mixed 2 0.5
```

### Available Modes
- `mixed`: Plasma base + all glitch effects, audio-driven (DEFAULT)
- `neural`: Plasma + VAE texture blend, then glitch effects
- `chaos`: Pure audio-driven plasma, no VAE, no glitch effects
- `drakonix`: Fursona palette stripes + Perlin-style noise + glitch effects
- `bitcrush`: Bit depth reduction only
- `rgb_split`: RGB channel separation only
- `datamosh`: Data corruption only

### Programmatic Usage
```python
from main import DoggyNeuralGlitch
from config import Config

glitcher = DoggyNeuralGlitch(Config())
glitcher.process(
    audio_path="song.mp3",
    output_dir="./output",
    mode="mixed",
    max_frames=50
)
```

---

## Design Philosophy

### Demoscene Principles
1. **Small**: Total codebase <1000 lines, model <5MB
2. **Fast**: 30 FPS generation on CPU (0.033s chunks); exports directly to MP4
3. **Cool**: Aesthetics over accuracy
4. **Hackable**: Easy to understand and modify
5. **Self-contained**: Minimal dependencies

### Why Random Weights Work
The neural core with random weights acts as a **consistent but chaotic hash function**:
- Similar audio features → similar latent codes
- Similar latent codes → similar (but unique) images  
- Nonlinear activations create emergent complexity
- No training needed - pure mathematical beauty
- Each audio chunk gets a unique but coherent frame

### Audio-Visual Coherence Philosophy
- **Low frequencies (bass)** → Structural changes (blocks, large displacements)
- **High frequencies (treble)** → Detail changes (RGB split, chromatic aberration)
- **Mid frequencies** → Tonal changes (bitcrush, datamosh)
- **Overall intensity** → Effect strength/amount
- **Spectral centroid** → Effect direction (horizontal vs vertical)

---

## Implementation Details

### Neural Core Implementation Notes

**Why This Architecture?**
- The **plasma generator** is the primary visual driver: audio features map directly to sine wave frequencies/phases, guaranteeing colorful and audio-reactive output with no training
- The **VAE acts as a texture layer** (blended 30% over plasma): its random weights with gain=2.0 produce complex nonlinear patterns that vary with the latent code
- Conv decoder (~400k params) stays demoscene-authentic — a fully-connected decoder would need 200M+ params for 512×512×3 output
- Tanh output naturally scales to image range with simple denormalization

**Key Implementation Tricks**:
1. **Feature padding**: MFCCs are padded/truncated to exactly 128 dimensions
2. **Normalization**: Features normalized to zero mean, unit variance
3. **Denormalization**: [-1,1] → [0,255] via: `((x + 1) * 127.5).clip(0, 255)`
4. **Seeding**: Fixed random seed (42069) for reproducibility

### Glitch Effects Implementation Notes

**Effect Order Matters**:
In mixed mode, effects are applied in this order:
1. Block glitches (if bass high)
2. RGB split (if treble high)  
3. Bitcrush (if mids high)
4. Chromatic aberration (always)
5. Pixel sorting (probabilistic)

**Why This Order?**
- Structural changes first (blocks)
- Detail changes second (RGB, chromatic)
- Tonal changes last (bitcrush)
- Sorting at end to work with corrupted data

**Datamoshing Technique**:
```python
flat = img.flatten()  # Convert to 1D
indices = random.choice(len(flat), n_corrupt)
flat[indices] = audio_bytes[:n_corrupt]
return flat.reshape(original_shape)
```

### Performance Optimizations

1. **No GPU**: Deliberately CPU-only for accessibility
2. **Batch size 1**: Single-frame generation reduces memory
3. **In-place operations**: NumPy operations reuse arrays when possible
4. **Minimal model**: ~400k parameters with conv decoder keeps inference fast
5. **Chunk processing**: Process one audio chunk at a time

---

## Testing

**test.py** generates synthetic audio and runs full pipeline:

```python
# Create 3-second test audio with:
# - Vibrato A note (440 Hz with modulation)
# - Higher vibrato (880 Hz)
# - Bass note (220 Hz)
# - White noise texture

# Process with:
# - 5 frames
# - Mixed mode
# - Output to ./test_output/

# Verifies entire pipeline works
```

---

## Extension Points

### Adding New Glitch Effects
1. Add static method to `GlitchEffects` class
2. Add mode case to `AudioDrivenGlitcher.apply_audio_driven_glitches()`
3. Update main.py argument parser choices

### Training the Neural Core
Currently uses random weights (which work great!), but can be trained:
```python
# VAE Loss
reconstruction_loss = mse_loss(output, target)
kl_divergence = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
loss = reconstruction_loss + beta * kl_divergence
```

### Adding Video Output
Use imageio or opencv to stitch frames:
```python
import imageio
frames = [Image.open(f) for f in sorted(frame_paths)]
imageio.mimsave('output.gif', frames, fps=30)
```

### Real-time Microphone Input
Use sounddevice for live audio:
```python
import sounddevice as sd
def callback(indata, frames, time, status):
    features = extract_features(indata)
    frame = generate_frame(features)
    display(frame)
```

---

## Known Limitations

1. **No GPU support**: Deliberately CPU-only
2. **Fixed image size**: 512×512 (configurable in config.py, not at runtime)
3. **No pre-trained weights**: Uses random initialization (intentional — see Design Philosophy)
4. **Single-threaded**: No parallel processing
5. **English-centric**: Comments and docs in English only

---

## Future Enhancements

- [ ] 1-bit quantized model (even smaller!)
- [x] MP4 output via justfile + ffmpeg
- [ ] GIF output built-in
- [ ] Live visualizer GUI
- [ ] MIDI controller support
- [ ] StyleGAN integration option
- [ ] GLSL shader export
- [ ] WebAssembly version
- [ ] Mobile app version

---

## Performance Benchmarks

**Tested on**: Modern x86_64 CPU (4 cores, 2.5GHz)

- Model initialization: <1 second
- Per-frame generation: ~0.03-0.1 seconds at 30fps (0.033s chunks)
- Generation rate: ~15-30 FPS observed on a modern x86_64 CPU
- Memory usage: 50-80 MB (steady state)
- Model size on disk: 2-4 MB

**Bottlenecks**:
1. Librosa feature extraction (~50% of time)
2. Plasma generation + VAE decode (~25% of time)
3. Glitch effects (~15% of time)
4. I/O (PNG save) (~10% of time)

---

## ASCII Art Banner

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  ██████╗  ██████╗  ██████╗  ██████╗ ██╗   ██╗             ║
║  ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝ ╚██╗ ██╔╝             ║
║  ██║  ██║██║   ██║██║  ███╗██║  ███╗ ╚████╔╝              ║
║  ██║  ██║██║   ██║██║   ██║██║   ██║  ╚██╔╝               ║
║  ██████╔╝╚██████╔╝╚██████╔╝╚██████╔╝   ██║                ║
║  ╚═════╝  ╚═════╝  ╚═════╝  ╚═════╝    ╚═╝                ║
║                                                           ║
║       N E U R A L   G L I T C H   E N G I N E             ║
║                   v0.1 - meltingscales                    ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Regeneration Instructions

To recreate this project from scratch using Claude:

1. Show Claude this README
2. Request: "Create DoggyNeuralGlitch based on this specification"
3. Claude will generate all files with proper structure
4. Test with: `uv run test.py`

Key points to emphasize:
- Demoscene philosophy (small, fast, cool)
- CPU-only operation
- Plasma generator as primary visual driver; VAE as texture layer (no training needed)
- Audio-driven effects with lowered thresholds for dense glitch output
- `drakonix` mode: fursona palette stripes + Perlin-style noise
- 30 FPS default; MP4 export via justfile + ffmpeg
- Conv decoder (~400k params, ~2-4MB, ~50-80MB RAM)

---

## License

MIT License - See LICENSE file

---

## Credits

Created by meltingscales in response to a social media post about extreme quantization enabling demoscene-style neural art.

*Because sometimes you need to quantize reality.*

🐕✨
