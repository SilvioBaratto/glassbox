"""Offline text tokenisation for Glassbox (issue #3).

Loads the CLIP BPE tokenizer once via HuggingFace, encodes a list of Italian
sentences, and saves:

  - ``data/tokens.npy``         — shape (N, 77), dtype int64
  - ``data/token_strings.json`` — per-position decoded sub-tokens
                                 (schema: ``{rows, model, max_len}``)

The output is the data backbone for scene 01 (tokenization) and scene 02
(LLM number pipeline). All inference happens here, offline; Manim scenes
load only the resulting NumPy / JSON.

Usage:
    python extract/extract_text.py                       # default: script.md
    python extract/extract_text.py --texts "gatto" "un cane"
    python extract/extract_text.py --out-dir /tmp/glass  # custom output dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from transformers import AutoTokenizer, BatchEncoding

try:
    from extract._common import parse_script_text
except ModuleNotFoundError:  # allow ``python extract/extract_text.py`` direct
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from _common import parse_script_text  # type: ignore[no-redef]

LOGGER = logging.getLogger("glassbox.extract_text")

# --- Constants -------------------------------------------------------------

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_MAX_LEN = 77
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "script.md"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"

# --- Tokenisation ----------------------------------------------------------


def _encode(
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_length: int = DEFAULT_MAX_LEN,
) -> BatchEncoding:
    """Run the tokenizer with CLIP's contract: max-length padding, int64 IDs.

    ``truncation=True`` is required alongside ``padding="max_length"`` for the
    fast tokenizer to build a uniform tensor when one or more inputs exceed
    ``max_length`` tokens. Without it, the call raises
    "Unable to create tensor ... excessive nesting".
    """
    return tokenizer(  # type: ignore[call-arg]
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _decode_row(tokenizer: AutoTokenizer, ids: Sequence[int]) -> list[str]:
    """Per-position sub-token strings, including special tokens.

    Returns a list of length ``len(ids)`` where each entry is the literal
    sub-token at that position (e.g. ``"<|startoftext|>"``, ``"gatto"``,
    ``"<|endoftext|>"``, ``"<|endoftext|>"``, ...). Empty string for IDs
    that do not map to a sub-token in the vocab.
    """
    # We rely on the public convert_ids_to_tokens when available; otherwise
    # fall back to per-id decode. The fake tokenizer in tests has neither.
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        try:
            return list(convert(list(ids)))
        except Exception:  # pragma: no cover — defensive
            pass
    out: list[str] = []
    for i in ids:
        try:
            out.append(tokenizer.decode([int(i)]))  # type: ignore[attr-defined]
        except Exception:
            out.append("")
    return out


# --- Public entry point ---------------------------------------------------


def run(
    *,
    texts: Sequence[str] | None,
    model_name: str,
    out_dir: Path,
    script_path: Path = DEFAULT_SCRIPT_PATH,
    max_length: int = DEFAULT_MAX_LEN,
) -> int:
    """Run the full extraction pipeline. Returns 0 on success, non-zero on error."""
    if texts is None:
        texts = parse_script_text(script_path)
    if not texts:
        LOGGER.error("No texts to tokenise (script.md empty after filter?)")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enc = _encode(tokenizer, texts, max_length=max_length)
    input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
    if input_ids.shape != (len(texts), max_length):
        LOGGER.error(
            "Unexpected token shape %s, expected (%d, %d)",
            input_ids.shape,
            len(texts),
            max_length,
        )
        return 2

    # Persist tokens
    tokens_path = out_dir / "tokens.npy"
    np.save(tokens_path, input_ids)

    # Persist decoded sub-tokens per position
    rows: list[list[str]] = [_decode_row(tokenizer, row.tolist()) for row in input_ids]
    sidecar = {
        "model": model_name,
        "max_len": max_length,
        "rows": rows,
    }
    strings_path = out_dir / "token_strings.json"
    strings_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stdout preview — first row, first 10 positions
    preview_ids = input_ids[0, :10].tolist()
    preview_decoded = rows[0][:10]
    LOGGER.info(
        "Wrote %s (shape %s, dtype %s)", tokens_path, input_ids.shape, input_ids.dtype
    )
    LOGGER.info("Wrote %s (rows=%d)", strings_path, len(rows))
    print("Preview (row 0, positions 0–9):")
    for i, (tid, dec) in enumerate(zip(preview_ids, preview_decoded)):
        print(f"  [{i}] id={tid:<6d}  decoded={dec!r}")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help=f"script.md to read when --texts is not given (default: {DEFAULT_SCRIPT_PATH})",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="explicit list of sentences to tokenise; if omitted, --script is used",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"max sequence length / padding (default: {DEFAULT_MAX_LEN})",
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
        texts=args.texts,
        model_name=args.model,
        out_dir=args.out_dir,
        script_path=args.script,
        max_length=args.max_length,
    )


# Re-export for tests that import ``mod.parse_script_text`` directly.
__all__ = ["parse_script_text", "run", "main"]


if __name__ == "__main__":
    sys.exit(main())
