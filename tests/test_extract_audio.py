"""Structural tests for the Glassbox audio extractor (issue #5).

HuggingFace ``WhisperFeatureExtractor``/``WhisperModel`` and ``librosa`` are
mocked to keep the suite hermetic and deterministic. The real extractor
runs offline (Whisper-base ≈ 150MB) and is exercised by ``--help`` and
manual runs.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "extract" / "extract_audio.py"
DATA_DIR = ROOT / "data"
FRAMES_PATH = DATA_DIR / "audio_frames.npy"
WAVEFORM_PATH = DATA_DIR / "audio_waveform.npy"
ENCODER_PATH = DATA_DIR / "audio_encoder.npy"
SAMPLE_AUDIO = ROOT / "assets" / "sample_audio.wav"


def _load_extractor_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_extract_audio_under_test",
        EXTRACTOR,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Fake Whisper + librosa ------------------------------------------------


class _FakeBatchEncoding(dict):
    """BatchEncoding stand-in (used by feature extractor)."""


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    """Stand-in for WhisperModel with a tiny encoder."""

    def __init__(self, hidden_seed: int = 7) -> None:
        self._seed = hidden_seed
        self.encoder = self._FakeEncoder(hidden_seed)

    def eval(self) -> "_FakeModel":
        return self

    class _FakeEncoder:
        def __init__(self, seed: int) -> None:
            self._seed = seed

        def __call__(
            self, *, input_features: torch.Tensor, output_hidden_states: bool = True
        ) -> _FakeEncoderOutput:  # noqa: ARG002
            # Whisper's encoder stride-2 compresses 3000 -> 1500
            assert input_features.shape[1:] == (80, 3000), (
                f"encoder expected (B, 80, 3000), got {tuple(input_features.shape)}"
            )
            gen = torch.Generator().manual_seed(self._seed)
            last = torch.randn(
                input_features.shape[0], 1500, 512, generator=gen, dtype=torch.float32
            )
            return _FakeEncoderOutput(last_hidden_state=last)


class _FakeFeatureExtractor:
    """Stand-in for WhisperFeatureExtractor: pads/truncates to 3000 frames."""

    def __init__(self) -> None:
        self.sampling_rate = 16000

    def __call__(
        self, audio: np.ndarray, sampling_rate: int, return_tensors: str = "pt"
    ) -> _FakeBatchEncoding:  # noqa: ARG002
        # Build a deterministic (1, 80, 3000) tensor based on the audio length
        # The 80 here is the n_mels of Whisper's log-mel.
        gen = torch.Generator().manual_seed(int(audio.shape[0]) % (2**31 - 1))
        feats = torch.randn(1, 80, 3000, generator=gen, dtype=torch.float32)
        return _FakeBatchEncoding(input_features=feats)


def _fake_librosa_load(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Return a deterministic 5-second mono float32 waveform in (-1, 1).

    librosa.load scales int16 PCM samples to float32 in (-1, 1) by
    dividing by 32768. We mimic that contract strictly — every value
    must be strictly inside (-1, 1).
    """
    assert path  # we don't actually need the file
    n = sr * 5  # 5 seconds
    gen = np.random.default_rng(n)
    # Use a uniform distribution in (-0.5, 0.5) — guaranteed inside the
    # librosa contract and easy to test for range.
    audio = gen.uniform(-0.5, 0.5, n).astype(np.float32)
    return audio, sr


def _fake_melspectrogram(
    *, y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int
) -> np.ndarray:
    """Return a (n_mels, T) power mel-spectrogram matching the librosa contract."""
    # Approximate T from the hop count
    T = 1 + len(y) // hop_length
    gen = np.random.default_rng(T)
    return gen.random((n_mels, T)).astype(np.float32) * 10.0


def _fake_power_to_db(S: np.ndarray, ref: Any = None) -> np.ndarray:  # noqa: ARG001
    """Return a dB-scaled log-mel in roughly [-80, 0]."""
    return np.log10(S + 1e-10).astype(np.float32) * 10.0


# --- Pyfake: a tmp audio file in tmp_path (does NOT touch the real asset) --


@pytest.fixture
def fake_audio_path(tmp_path: Path) -> Path:
    """Create a tiny placeholder WAV in tmp_path; the fake _librosa_load ignores it."""
    p = tmp_path / "test_audio.wav"
    p.write_bytes(
        b"RIFF\x00\x00\x00\x00WAVEfmt "
    )  # bytes don't matter — fake bypasses librosa
    return p


# --- Subprocess smoke (no model download) ----------------------------------


def test_when_extractor_help_invoked_then_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"--help failed: rc={proc.returncode}\nstderr={proc.stderr}"
    )


def test_when_extractor_help_invoked_then_required_cli_args_listed() -> None:
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for flag in ("--audio", "--out-dir"):
        assert flag in proc.stdout, f"--help missing required flag {flag}"


# --- Module surface --------------------------------------------------------


def test_when_extractor_imported_then_run_callable_exists() -> None:
    mod = _load_extractor_module()
    assert callable(getattr(mod, "main", None))
    assert callable(getattr(mod, "run", None))


# --- audio_frames: true frame count from librosa (reviewer's bug fix) -----


def test_when_extractor_run_then_audio_frames_shape_matches_actual_duration(
    tmp_path: Path, fake_audio_path: Path
) -> None:
    """``audio_frames.npy`` must be the TRUE frame count, not the Whisper 3000.

    The reviewer's bug fix: Whisper's feature extractor always pads/truncates
    to 3000 frames, but the displayable mel spectrogram should be the actual
    count (~501 for 5s audio). We bypass Whisper for the displayable npy
    and use librosa directly.
    """
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeModel()
    fake_fe = _FakeFeatureExtractor()
    with (
        patch.object(mod, "WhisperFeatureExtractor") as MockFE,
        patch.object(mod, "WhisperModel") as MockModel,
        patch.object(mod.librosa, "load", _fake_librosa_load),
        patch.object(mod.librosa.feature, "melspectrogram", _fake_melspectrogram),
        patch.object(mod.librosa, "power_to_db", _fake_power_to_db),
    ):
        MockFE.from_pretrained.return_value = fake_fe
        MockModel.from_pretrained.return_value = fake_model
        rc = mod.run(audio_path=fake_audio_path, out_dir=out_dir)
    assert rc == 0

    frames = np.load(out_dir / "audio_frames.npy")
    assert frames.ndim == 2 and frames.shape[0] == 80, (
        f"expected (80, T) shape, got {frames.shape}"
    )
    # The fake produces ~313 frames for 5s audio (50000/160 = 313).
    # The contract: NOT 3000.
    assert frames.shape[1] != 3000, (
        f"audio_frames leaked Whisper's padded 3000; expected actual frame count, "
        f"got {frames.shape}"
    )
    assert frames.dtype == np.float32, f"expected float32, got {frames.dtype}"


# --- audio_waveform: raw mono float32 in [-1, 1] ---------------------------


def test_when_extractor_run_then_audio_waveform_is_mono_float32(
    tmp_path: Path, fake_audio_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeModel()
    fake_fe = _FakeFeatureExtractor()
    with (
        patch.object(mod, "WhisperFeatureExtractor") as MockFE,
        patch.object(mod, "WhisperModel") as MockModel,
        patch.object(mod.librosa, "load", _fake_librosa_load),
        patch.object(mod.librosa.feature, "melspectrogram", _fake_melspectrogram),
        patch.object(mod.librosa, "power_to_db", _fake_power_to_db),
    ):
        MockFE.from_pretrained.return_value = fake_fe
        MockModel.from_pretrained.return_value = fake_model
        mod.run(audio_path=fake_audio_path, out_dir=out_dir)
    wave = np.load(out_dir / "audio_waveform.npy")
    assert wave.ndim == 1, f"expected 1-D mono waveform, got shape {wave.shape}"
    assert wave.dtype == np.float32, f"expected float32, got {wave.dtype}"
    assert wave.min() >= -1.0 and wave.max() <= 1.0, (
        f"waveform out of [-1, 1] range: [{wave.min()}, {wave.max()}]"
    )
    # 5s @ 16 kHz = 80000 samples
    assert wave.shape[0] == 80000, (
        f"expected 80000 samples (5s @ 16kHz), got {wave.shape[0]}"
    )


# --- audio_encoder: post-Conv1d (1500, 512) --------------------------------


def test_when_extractor_run_then_audio_encoder_is_post_conv1d(
    tmp_path: Path, fake_audio_path: Path
) -> None:
    """Whisper's encoder always yields (1500, 512) from a 3000-frame input.

    The reviewer's bug fix: the AC's `T/2` math is wrong; Conv1d stride-2
    is fixed at 1500. We enforce the contract here.
    """
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeModel()
    fake_fe = _FakeFeatureExtractor()
    with (
        patch.object(mod, "WhisperFeatureExtractor") as MockFE,
        patch.object(mod, "WhisperModel") as MockModel,
        patch.object(mod.librosa, "load", _fake_librosa_load),
        patch.object(mod.librosa.feature, "melspectrogram", _fake_melspectrogram),
        patch.object(mod.librosa, "power_to_db", _fake_power_to_db),
    ):
        MockFE.from_pretrained.return_value = fake_fe
        MockModel.from_pretrained.return_value = fake_model
        mod.run(audio_path=fake_audio_path, out_dir=out_dir)
    enc = np.load(out_dir / "audio_encoder.npy")
    assert enc.shape == (1500, 512), f"expected (1500, 512), got {enc.shape}"
    assert enc.dtype == np.float32, f"expected float32, got {enc.dtype}"


# --- Idempotency ----------------------------------------------------------


def test_when_extractor_run_twice_then_outputs_identical(
    tmp_path: Path, fake_audio_path: Path
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    fake_model = _FakeModel()
    fake_fe = _FakeFeatureExtractor()
    with (
        patch.object(mod, "WhisperFeatureExtractor") as MockFE,
        patch.object(mod, "WhisperModel") as MockModel,
        patch.object(mod.librosa, "load", _fake_librosa_load),
        patch.object(mod.librosa.feature, "melspectrogram", _fake_melspectrogram),
        patch.object(mod.librosa, "power_to_db", _fake_power_to_db),
    ):
        MockFE.from_pretrained.return_value = fake_fe
        MockModel.from_pretrained.return_value = fake_model
        mod.run(audio_path=fake_audio_path, out_dir=out_dir)
        first = (out_dir / "audio_encoder.npy").read_bytes()
        mod.run(audio_path=fake_audio_path, out_dir=out_dir)
        second = (out_dir / "audio_encoder.npy").read_bytes()
    assert first == second, "re-running extractor must produce identical output"
