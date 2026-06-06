# Sample assets

Both files in this directory are **programmatically generated** by
[`scripts/generate_assets.py`](../scripts/generate_assets.py) and committed to
the repository so that extraction scripts (issues #3–#6) and Manim scenes
have deterministic input on every machine.

## Files

### `sample_image.jpg`

| Property | Value |
|----------|-------|
| Size | 224×224 |
| Mode | RGB |
| Format | JPEG (quality 80, optimized) |
| File size | ~6 KB |
| Source | Synthesised by Pillow in `scripts/generate_assets.py` |
| License | MIT (this project) |

**Content.** A stylised landscape built from layered shapes:

- vertical sky gradient (blue → warm horizon)
- yellow sun disc with soft halo (upper-right)
- two overlapping mountain silhouettes with snow caps
- green ground strip with deterministic grass strokes
- a black cat silhouette with yellow eyes (foreground)
- the Italian caption `gatto` in the lower-left

**Why not a real photograph.** A real cat photo would be aesthetically
nicer, but adds three problems:

1. **License ambiguity.** A public-domain JPEG still requires attribution
   and provenance. A synthesised image is unambiguously MIT.
2. **Network dependency at build time.** We refuse to download arbitrary
   data when extraction scripts run; a deterministic generator avoids
   that.
3. **ViT-16 patch quality.** A real photograph would still satisfy the
   visual-content test, but a layered scene with hard edges (cat
   silhouette, snow caps, sun disc) gives the ViT-16 patch embeddings
   (issue #4) explicit structure to extract — the same structure the
   final educational video will narrate.

**Determinism.** The image is rendered with `np.random.default_rng(20260606)`;
re-running `python scripts/generate_assets.py` with the same seed produces a
byte-identical JPEG (subject to PIL's internal hash randomness in optimisations,
which we set explicitly to deterministic by re-running with `optimize=True`).

### `sample_audio.wav`

| Property | Value |
|----------|-------|
| Channels | 1 (mono) |
| Sample rate | 16 000 Hz |
| Sample width | 16-bit PCM (PCM_16) |
| Duration | 5.0 s |
| File size | ~156 KB |
| Source | Synthesised by NumPy in `scripts/generate_assets.py` |
| License | MIT (this project) |

**Content.** A 5-second mono signal resembling speech-like spectral content:

- four formants at 440, 1200, 2400, 3500 Hz (vowel-shaped harmonics)
- a 4 Hz syllabic amplitude envelope
- additive pink-ish noise (-30 dB equivalent) to exercise Whisper's
  feature extractor the way real speech does

**Why not a real recording.** Same three reasons as the image, plus a
technical one: a pure 440 Hz tone (the issue's first suggestion) would
not exercise Whisper's mel-spectrogram pipeline well — a 5-second pure
sine produces degenerate log-mel frames. The synthesised variant gives
the audio chunking scene (issue #4) meaningful visual structure.

**Why PCM_16 explicitly.** The issue review flagged that `soundfile.write`
defaults to `FLOAT` subtype, which would push the file size to ~640 KB
and break the < 1 MB budget in a different way than just the duration
multiplier. We use the Python stdlib `wave` module to set
`setsampwidth(2)` and `PCM_16` directly, bypassing any library default.

## Provenance

Both files are produced by a single script with a fixed seed. To verify or
regenerate:

```bash
python scripts/generate_assets.py
```

This is idempotent and safe to re-run. To produce a fresh seed:

```bash
python scripts/generate_assets.py --seed 42
```

## License

Both files are part of this project and released under the **MIT License**
(see [`../LICENSE`](../LICENSE)). They contain no third-party content.
