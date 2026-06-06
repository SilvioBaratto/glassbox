"""Structural tests for the Glassbox bootstrap (issue #1).

These tests verify that the project skeleton, requirements, and render.sh
contract hold. They are intentionally light — the visual output itself is
the smoke test (`bash render.sh _smoke`).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --- Acceptance: directory layout -------------------------------------------


@pytest.mark.parametrize("subdir", ["extract", "data", "scenes", "assets"])
def test_when_project_layout_requested_subdir_exists_then_path_is_directory(
    subdir: str,
) -> None:
    target = ROOT / subdir
    assert target.is_dir(), f"Missing project directory: {target}"


@pytest.mark.parametrize("subdir", ["extract", "data", "scenes", "assets"])
def test_when_subdir_created_then_gitkeep_keeps_it_tracked(subdir: str) -> None:
    assert (ROOT / subdir / ".gitkeep").exists(), (
        f"{subdir}/.gitkeep missing — directory would not be tracked by git"
    )


# --- Acceptance: requirements.txt pins the expected packages -----------------

EXPECTED_PINS = {
    "manim",
    "manim-ml",
    "transformers",
    "torch",
    "scikit-learn",
    "librosa",
    "numpy",
    "matplotlib",
    "pillow",
}


def test_when_requirements_read_then_all_required_packages_pinned() -> None:
    content = (ROOT / "requirements.txt").read_text().lower()
    for pkg in EXPECTED_PINS:
        # Match:  <pkg> [version specifier chars like ==, >=, <, <=, ~=, !=]
        assert re.search(rf"^{re.escape(pkg)}\s*[=<>!~]", content, re.MULTILINE), (
            f"requirements.txt missing pinned entry for: {pkg}"
        )


def test_when_transformers_pinned_then_minimum_supports_output_hidden_states() -> None:
    content = (ROOT / "requirements.txt").read_text()
    match = re.search(r"^transformers\s*>=\s*([\d.]+)", content, re.MULTILINE)
    assert match, "transformers must be pinned with a lower bound"
    assert tuple(int(x) for x in match.group(1).split(".")[:2]) >= (4, 40), (
        "transformers >= 4.40 required for stable output_hidden_states / "
        "CLIPModel.image_embeds API"
    )


# --- Acceptance: .gitignore excludes artefacts --------------------------------

EXPECTED_IGNORES = [
    r"data/.*\.npy",
    r"^output/",
    r"^\.venv\s*$",  # `.venv` on its own line (existing entry)
    r"__pycache__/",
    r"\.mp4$",
    r"checkpoints/",
]


def test_when_gitignore_read_then_all_artefact_patterns_present() -> None:
    content = (ROOT / ".gitignore").read_text()
    for pattern in EXPECTED_IGNORES:
        assert re.search(pattern, content, re.MULTILINE), (
            f".gitignore missing pattern matching: {pattern}"
        )


def test_when_script_md_is_source_of_truth_then_it_is_force_tracked() -> None:
    """script.md is the narration source — must not be ignored."""
    content = (ROOT / ".gitignore").read_text()
    assert "!script.md" in content, (
        "script.md is the project's source of truth and must be force-added"
    )


# --- Acceptance: render.sh argument contract ---------------------------------


def test_when_render_sh_called_with_no_args_then_exits_nonzero() -> None:
    proc = subprocess.run(
        ["bash", str(ROOT / "render.sh")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, "render.sh must reject empty argument list"
    assert "Usage" in proc.stderr or "Usage" in proc.stdout


def test_when_render_sh_called_with_missing_scene_then_exits_with_code_3() -> None:
    proc = subprocess.run(
        ["bash", str(ROOT / "render.sh"), "this_scene_does_not_exist"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3, (
        f"Expected exit code 3 for missing scene file, got {proc.returncode}: "
        f"{proc.stderr}"
    )


def test_when_render_sh_called_with_bad_quality_then_exits_with_code_5() -> None:
    proc = subprocess.run(
        ["bash", str(ROOT / "render.sh"), "_smoke", "ultra"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 5, (
        f"Expected exit code 5 for unknown quality, got {proc.returncode}"
    )


def test_when_render_sh_set_strict_mode_then_uses_euo_pipefail() -> None:
    content = (ROOT / "render.sh").read_text()
    assert "set -euo pipefail" in content


def test_when_render_sh_executed_then_output_directory_is_created() -> None:
    """Side effect: output/ should exist after any successful render setup.

    We don't run manim (not installed in test env); we exercise the pre-flight
    by giving an unknown quality which exits BEFORE the mkdir in some shells,
    so we instead check that the script is *structured* to mkdir output/."""
    content = (ROOT / "render.sh").read_text()
    assert "mkdir -p output" in content, "render.sh must create output/ before manim"
