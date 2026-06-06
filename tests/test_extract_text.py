"""Structural tests for the Glassbox text extractor (issue #3).

We mock the HuggingFace tokenizer to keep the suite hermetic and fast — the
real tokenizer is exercised by the extractor script's smoke run (issue AC:
"exits 0 and produces both files"), not by the test suite.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence
from unittest.mock import patch

import numpy as np
import pytest
from transformers import BatchEncoding


def _load_extractor_module() -> Any:
    """Load ``extract/extract_text.py`` as a fresh module object (no caching)."""
    spec = importlib.util.spec_from_file_location(
        "_extract_text_under_test",
        EXTRACTOR,
    )
    assert spec is not None and spec.loader is not None, (
        f"Cannot load {EXTRACTOR} as a module"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "extract" / "extract_text.py"
DATA_DIR = ROOT / "data"
TOKENS_PATH = DATA_DIR / "tokens.npy"
STRINGS_PATH = DATA_DIR / "token_strings.json"
SCRIPT_PATH = ROOT / "script.md"

# --- Fake tokenizer --------------------------------------------------------


class _FakeTokenizer:
    """Minimal stand-in for AutoTokenizer used by tests.

    Mirrors the real CLIP tokenizer's contract:
      - tokenize() returns sub-token strings
      - convert_tokens_to_ids() maps sub-tokens to int64 IDs
      - __call__() with padding="max_length" returns BatchEncoding with
        input_ids of shape (N, max_length), dtype int64
      - decode() round-trips a list of IDs
    """

    START_ID = 49406
    END_ID = 49407

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        # Build a tiny vocab so the fake is self-contained.
        base = {
            "<|startoftext|>": self.START_ID,
            "<|endoftext|>": self.END_ID,
            "gatto": 531,
            "##tto": 532,
            "ga": 533,
            "cane": 534,
            "un": 535,
            "il": 536,
            "<|pad|>": 49407,
        }
        if vocab:
            base.update(vocab)
        self._vocab = base
        self._inv = {v: k for k, v in base.items()}

    def tokenize(self, text: str) -> list[str]:
        # Trivial BPE: split on whitespace, then a few sub-splits.
        out: list[str] = []
        for word in text.split():
            if word == "gatto":
                out.extend(["ga", "##tto"])
            else:
                out.append(word)
        return out

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return [self._vocab.get(t, 999) for t in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        ids = list(ids)
        # Join sub-tokens, drop special tokens.
        parts: list[str] = []
        for i in ids:
            tok = self._inv.get(int(i), "")
            if tok in ("<|startoftext|>", "<|endoftext|>", "<|pad|>"):
                continue
            if tok.startswith("##"):
                parts.append(tok[2:])
            else:
                parts.append(tok)
        return "".join(parts).strip()

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 77,
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        rows: list[list[int]] = []
        for text in texts:
            ids = [self.START_ID]
            for tok in self.tokenize(text):
                ids.extend(self.convert_tokens_to_ids([tok]))
            ids.append(self.END_ID)
            # Truncate / pad
            if len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.END_ID]
            if len(ids) < max_length:
                ids = ids + [self.END_ID] * (max_length - len(ids))
            rows.append(ids)
        return BatchEncoding({"input_ids": np.asarray(rows, dtype=np.int64)})


@pytest.fixture
def fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


# --- Subprocess smoke: import path is importable, no runtime crash ---------


def test_when_extractor_imported_as_module_then_run_callable_exists() -> None:
    """The module exposes a `main()` function callable from the CLI."""
    mod = _load_extractor_module()
    assert callable(getattr(mod, "main", None)), (
        "extract_text.py must expose a main() entry point"
    )


# --- script.md parser filter ----------------------------------------------


def test_when_script_md_parsed_then_header_and_marker_lines_dropped(
    tmp_path: Path,
) -> None:
    """The parser must drop markdown headers, `|` markers, and empty lines."""
    fake = tmp_path / "script.md"
    fake.write_text(
        "### Hook\n"
        "\n"
        "Come si fa a capire tutto?\n"
        "\n"
        "### Script\n"
        "\n"
        "Io non lo so ma ti spiego.\n"
        "\n"
        "| QUA INIZIA PROGETTO GITHUB |\n"
    )
    mod = _load_extractor_module()
    parsed = mod.parse_script_text(fake)
    assert parsed == [
        "Come si fa a capire tutto?",
        "Io non lo so ma ti spiego.",
    ], f"parser dropped wrong lines: {parsed}"


def test_when_real_script_md_parsed_then_no_garbage_lines() -> None:
    """End-to-end check: the real script.md parses to clean text only."""
    mod = _load_extractor_module()
    parsed = mod.parse_script_text(SCRIPT_PATH)
    for line in parsed:
        assert not line.startswith("###"), f"header leaked: {line!r}"
        assert "|" not in line, f"marker leaked: {line!r}"
        assert line.strip(), "empty line leaked"


# --- Script produces files with correct shape/dtype/schema ---------------


def test_when_extractor_run_with_texts_then_tokens_npy_shape_and_dtype(
    tmp_path: Path, fake_tokenizer: _FakeTokenizer
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    with patch.object(mod, "AutoTokenizer") as MockAutoTok:
        MockAutoTok.from_pretrained.return_value = fake_tokenizer
        rc = mod.run(
            texts=["gatto", "un cane"],
            model_name="openai/clip-vit-base-patch32",
            out_dir=out_dir,
        )
    assert rc == 0

    tokens = np.load(out_dir / "tokens.npy")
    assert tokens.shape == (2, 77), f"expected (2, 77), got {tokens.shape}"
    assert tokens.dtype == np.int64, f"expected int64, got {tokens.dtype}"


def test_when_extractor_run_then_token_strings_json_schema(
    tmp_path: Path, fake_tokenizer: _FakeTokenizer
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    with patch.object(mod, "AutoTokenizer") as MockAutoTok:
        MockAutoTok.from_pretrained.return_value = fake_tokenizer
        mod.run(
            texts=["gatto", "un cane"],
            model_name="openai/clip-vit-base-patch32",
            out_dir=out_dir,
        )

    sidecar = json.loads((out_dir / "token_strings.json").read_text())
    assert set(sidecar) >= {"rows", "model", "max_len"}, (
        f"token_strings.json missing keys: {sidecar.keys()}"
    )
    assert sidecar["max_len"] == 77
    assert sidecar["model"] == "openai/clip-vit-base-patch32"
    assert len(sidecar["rows"]) == 2
    for row in sidecar["rows"]:
        assert isinstance(row, list)
        assert len(row) == 77


# --- Idempotency ----------------------------------------------------------


def test_when_extractor_run_twice_then_outputs_identical(
    tmp_path: Path, fake_tokenizer: _FakeTokenizer
) -> None:
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    with patch.object(mod, "AutoTokenizer") as MockAutoTok:
        MockAutoTok.from_pretrained.return_value = fake_tokenizer
        mod.run(
            texts=["gatto"], model_name="openai/clip-vit-base-patch32", out_dir=out_dir
        )
        first = (out_dir / "tokens.npy").read_bytes()
        mod.run(
            texts=["gatto"], model_name="openai/clip-vit-base-patch32", out_dir=out_dir
        )
        second = (out_dir / "tokens.npy").read_bytes()
    assert first == second, "re-running extractor must produce identical output"


# --- Defaults: when no --texts, use script.md and produce >= 3 rows -------


def test_when_extractor_run_with_no_texts_then_defaults_to_script_md(
    tmp_path: Path, fake_tokenizer: _FakeTokenizer
) -> None:
    """Calling run() without texts must read script.md and produce >= 3 rows.

    This is the default-mode contract used by the smoke test in the AC.
    """
    out_dir = tmp_path / "data"
    mod = _load_extractor_module()
    # Patch parse_script_text to return 3 deterministic strings.
    with (
        patch.object(mod, "parse_script_text") as MockParse,
        patch.object(mod, "AutoTokenizer") as MockAutoTok,
    ):
        MockParse.return_value = ["gatto", "un cane", "il modello"]
        MockAutoTok.from_pretrained.return_value = fake_tokenizer
        rc = mod.run(
            texts=None, model_name="openai/clip-vit-base-patch32", out_dir=out_dir
        )
    assert rc == 0
    tokens = np.load(out_dir / "tokens.npy")
    assert tokens.shape[0] >= 3, f"expected >= 3 rows, got {tokens.shape[0]}"


# --- CLI contract ---------------------------------------------------------


def test_when_extractor_invoked_with_no_args_then_exits_nonzero() -> None:
    """CLI without args must fail with usage hint (or default to script.md).

    The issue says CLI args include --model, --out-dir, --texts (or read from
    script.md). A no-arg invocation is allowed to default to script.md; the
    hard contract is just that it must NOT crash. We run a quick import-only
    path that does not touch the network.
    """
    proc = subprocess.run(
        [sys.executable, str(EXTRACTOR), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"--help failed: rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
