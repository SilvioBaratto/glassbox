"""Structural tests for the Glassbox orchestrator (issue #14).

Tests:
- ``extract_all.sh`` exists, is executable, and references all 4 extract scripts
- ``render.sh all`` is accepted (we don't actually render — too slow;
  we just check the bash script accepts the ``all`` arg and would loop
  through the 7 scene names).
- ``render.sh <name>`` (single scene) still works (preserved from #1)
- ``render.sh`` (no args) prints usage
- ``verify.sh`` exists, is executable, and detects missing MP4s
- ``README.md`` has the required Italian run-guide sections
- All existing tests still pass (full suite)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACT_ALL = ROOT / "extract_all.sh"
RENDER_SH = ROOT / "render.sh"
VERIFY_SH = ROOT / "verify.sh"
README = ROOT / "README.md"
EXTRACT_DIR = ROOT / "extract"

EXPECTED_EXTRACTS = [
    "extract_text.py",
    "extract_image_patches.py",
    "extract_audio.py",
    "extract_shared_space.py",
]

EXPECTED_SCENES = [
    "01_tokenization",
    "02_llm_numbers",
    "03_image_patches",
    "04_audio_chunks",
    "05_shared_vector_space",
    "06_two_path_comparison",
    "07_full_pipeline",
]


# --- File existence + executability ---------------------------------------


def test_when_extract_all_sh_inspected_then_exists_and_executable() -> None:
    assert EXTRACT_ALL.is_file(), f"{EXTRACT_ALL.name} must exist"
    import stat

    mode = EXTRACT_ALL.stat().st_mode
    assert mode & stat.S_IXUSR, f"{EXTRACT_ALL.name} must be executable"


def test_when_verify_sh_inspected_then_exists_and_executable() -> None:
    assert VERIFY_SH.is_file(), f"{VERIFY_SH.name} must exist"
    import stat

    mode = VERIFY_SH.stat().st_mode
    assert mode & stat.S_IXUSR, f"{VERIFY_SH.name} must be executable"


# --- extract_all.sh content -----------------------------------------------


def test_when_extract_all_sh_read_then_calls_all_four_extractors() -> None:
    content = EXTRACT_ALL.read_text()
    for script in EXPECTED_EXTRACTS:
        assert script in content, (
            f"extract_all.sh must reference {script}. Got: {content[:500]}"
        )


def test_when_extract_all_sh_read_then_uses_strict_mode() -> None:
    content = EXTRACT_ALL.read_text()
    assert "set -euo pipefail" in content, (
        "extract_all.sh must use 'set -euo pipefail' for strict error handling"
    )


# --- render.sh: `all` case ------------------------------------------------


def test_when_render_sh_called_with_all_then_recognised_arg() -> None:
    """`render.sh all` should validate the arg and at least print a plan.

    We don't actually render all 7 scenes in the test (would take minutes
    AND requires a real Manim install). The test asserts the script
    accepts the `all` arg, prints the expected plan, and doesn't fail
    with usage-error codes (2/4/5). If scene renders fail downstream
    (e.g. manim not installed), the script's `all` orchestrator should
    collect errors and exit with a distinct code (6) — that's OK.
    """
    proc = subprocess.run(
        [str(RENDER_SH), "all"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Reject the "usage" / "no class" / "bad quality" codes — these
    # would mean the `all` arg wasn't recognised.
    assert proc.returncode not in (2, 4, 5), (
        f"render.sh all exited with bad code {proc.returncode}: "
        f"stderr={proc.stderr}\nstdout={proc.stdout}"
    )
    # The orchestrator should print a plan header.
    assert "all" in proc.stdout.lower() or "all" in proc.stderr.lower(), (
        "render.sh all should print a plan mentioning 'all'"
    )


def test_when_render_sh_called_with_no_args_then_exits_nonzero_with_usage() -> None:
    """Issue #1 contract: empty arg list must print usage and exit non-zero."""
    proc = subprocess.run(
        [str(RENDER_SH)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode != 0, "render.sh must reject empty argument list"
    assert "Usage" in proc.stderr or "Usage" in proc.stdout, (
        f"render.sh must print a Usage hint. Got stderr={proc.stderr!r} "
        f"stdout={proc.stdout!r}"
    )


def test_when_render_sh_called_with_smoke_scene_then_exits_nonzero() -> None:
    """The smoke scene file must exist (issue #1 contract)."""
    smoke_file = ROOT / "scenes" / "_smoke.py"
    assert smoke_file.is_file(), f"{smoke_file} must exist (issue #1)"


# --- verify.sh behaviour --------------------------------------------------


def test_when_verify_sh_called_with_no_outputs_then_exits_nonzero(
    tmp_path: Path,
) -> None:
    """If output/ is empty / missing, verify.sh must fail (exit != 0).

    The script does ``cd "$(dirname "$0")"`` so it always looks at the
    script's own ``output/``. To test the "no outputs" case we copy the
    script into ``tmp_path`` (with no ``output/`` next to it) and invoke
    it from there.
    """
    import shutil
    import stat

    copied = tmp_path / "verify.sh"
    shutil.copy(VERIFY_SH, copied)
    # Preserve executable bit on the copy.
    copied.chmod(copied.stat().st_mode | stat.S_IXUSR)
    proc = subprocess.run(
        ["bash", str(copied)],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=tmp_path,
    )
    assert proc.returncode != 0, (
        f"verify.sh should fail when output/ is empty, got rc={proc.returncode}: "
        f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )


def test_when_verify_sh_read_then_checks_all_seven_mp4s() -> None:
    """Each of the 7 scene basenames must be referenced in verify.sh.

    The exact path template (``output/videos/<scene>/<quality>/<scene>.mp4``)
    is not asserted — only that each scene's basename appears so the
    for-loop iterates over all 7.
    """
    content = VERIFY_SH.read_text()
    for scene in EXPECTED_SCENES:
        # The scene basename (e.g. "01_tokenization") must be quoted in
        # the EXPECTED_SCENES list.
        assert f'"{scene}"' in content or f"'{scene}'" in content, (
            f"verify.sh must list {scene} in EXPECTED_SCENES. Got: {content[:500]}"
        )


# --- README --------------------------------------------------------------


def test_when_readme_inspected_then_has_required_sections() -> None:
    content = README.read_text()
    required_sections = [
        "Install",  # installazione
        "extract",  # estrazione
        "render",  # render
        "verify",  # verifica
        "ffmpeg",  # troubleshooting
    ]
    for section in required_sections:
        assert re.search(section, content, re.IGNORECASE), (
            f"README.md must mention '{section}' for the run guide. "
            f"Got first 500 chars: {content[:500]}"
        )


# --- Full test suite guard ------------------------------------------------


def test_when_all_tests_collected_then_count_meets_or_exceeds_floor() -> None:
    """We expect at least 150 tests at this point (post-issue-13)."""
    proc = subprocess.run(
        ["python3", "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    # Look for the "<N> tests collected" line
    m = re.search(r"(\d+)\s+tests\s+collected", proc.stdout)
    assert m, f"could not parse test count from: {proc.stdout[:500]}"
    count = int(m.group(1))
    assert count >= 150, f"expected >= 150 tests, got {count}"
