"""Offline audio frame extraction for Glassbox (issue #5).

Produces three artifacts in ``data/`` from the sample WAV:

  - ``audio_frames.npy``   — log-mel spectrogram, shape (80, T_actual), float32.
                            Computed via librosa directly (not the Whisper
                            feature extractor) so the displayed mel count
                            matches the actual audio duration. The Whisper
                            extractor pads to 3000 frames regardless of
                            input length; that would mislead the visual.
  - ``audio_waveform.npy`` — raw mono float32 waveform, range [-1, 1], shape
                            (N,) where N = duration * 16000.
  - ``audio_encoder.npy`` — Whisper encoder last_hidden_state after the
                            Conv1d stride-2 stem, shape (1500, 512), float32.
                            Always 1500 because the encoder input is fixed
                            at 3000 frames (Whisper contract).

Usage:
    python extract/extract_audio.py                       # default
    python extract/extract_audio.py --audio path/to.wav
    python extract/extract_audio.py --out-dir /tmp/glass
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

LOGGER = logging.getLogger("glassbox.extract_audio")

# --- Constants -------------------------------------------------------------

DEFAULT_MODEL = "openai/whisper-base"
DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "assets" / "sample_audio.wav"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
# Whisper's fixed input length (encoder contract)
WHISPER_FIXED_FRAMES = 3000
# After the Conv1d stride-2 stem
ENCODER_POSITIONS = WHISPER_FIXED_FRAMES // 2  # 1500
ENCODER_HIDDEN = 512


# --- Loader helpers --------------------------------------------------------


def _load_audio(audio_path: Path) -> np.ndarray:
    """Read the WAV as mono float32 at 16 kHz.

    ``librosa.load`` returns float32 in [-1, 1] by default; that's the
    format downstream scene 04 expects (Manim plots amplitudes directly).
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")
    return audio.astype(np.float32, copy=False)


def _compute_log_mel(audio: np.ndarray) -> np.ndarray:
    """Compute the displayable log-mel spectrogram via librosa directly.

    The number of frames depends on the audio duration; for our 5-second
    sample at hop_length=160, this is ~501 frames (not 3000).
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


# --- Model forward ---------------------------------------------------------


def _whisper_input_features(
    feature_extractor: WhisperFeatureExtractor, audio: np.ndarray
) -> torch.Tensor:
    """Run the Whisper feature extractor — returns (1, 80, 3000) padded."""
    enc = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return enc["input_features"]


def _run_encoder(model: WhisperModel, input_features: torch.Tensor) -> np.ndarray:
    """Run WhisperModel.encoder — returns (1500, 512) numpy array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            out = model.encoder(
                input_features=input_features, output_hidden_states=True
            )
    arr = out.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return arr


# --- Public entry point ---------------------------------------------------


def run(
    *,
    audio_path: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
) -> int:
    """Run the full extraction pipeline. Returns 0 on success, non-zero on error."""
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = _load_audio(audio_path)

    # Displayable mel (true frame count) — bypasses Whisper padding
    log_mel = _compute_log_mel(audio)
    np.save(out_dir / "audio_frames.npy", log_mel)

    # Raw waveform
    np.save(out_dir / "audio_waveform.npy", audio)

    # Whisper encoder hidden state (fixed 1500 positions)
    LOGGER.info("Loading Whisper feature extractor: %s", model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    LOGGER.info("Loading Whisper model: %s", model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = WhisperModel.from_pretrained(model_name).eval()

    input_features = _whisper_input_features(feature_extractor, audio)
    if input_features.shape != (1, N_MELS, WHISPER_FIXED_FRAMES):
        LOGGER.error(
            "Unexpected input_features shape %s, expected (1, %d, %d)",
            tuple(input_features.shape),
            N_MELS,
            WHISPER_FIXED_FRAMES,
        )
        return 2
    encoder_hidden = _run_encoder(model, input_features)
    if encoder_hidden.shape != (ENCODER_POSITIONS, ENCODER_HIDDEN):
        LOGGER.error(
            "Unexpected encoder hidden shape %s, expected (%d, %d)",
            encoder_hidden.shape,
            ENCODER_POSITIONS,
            ENCODER_HIDDEN,
        )
        return 3
    np.save(out_dir / "audio_encoder.npy", encoder_hidden)

    # Stdout preview
    LOGGER.info(
        "Wrote %s (shape %s, dtype %s)",
        out_dir / "audio_frames.npy",
        log_mel.shape,
        log_mel.dtype,
    )
    LOGGER.info(
        "Wrote %s (shape %s, dtype %s)",
        out_dir / "audio_waveform.npy",
        audio.shape,
        audio.dtype,
    )
    LOGGER.info(
        "Wrote %s (shape %s, dtype %s)",
        out_dir / "audio_encoder.npy",
        encoder_hidden.shape,
        encoder_hidden.dtype,
    )
    print(f"audio_frames.npy:   shape={log_mel.shape} dtype={log_mel.dtype}")
    print(f"audio_waveform.npy: shape={audio.shape} dtype={audio.dtype}")
    print(
        f"audio_encoder.npy:  shape={encoder_hidden.shape} dtype={encoder_hidden.dtype}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help=f"input WAV (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable DEBUG logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return run(
        audio_path=args.audio,
        out_dir=args.out_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    sys.exit(main())
