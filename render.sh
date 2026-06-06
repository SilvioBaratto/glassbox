#!/usr/bin/env bash
#
# render.sh — render a single Manim scene (or all 7) into output/.
#
# Usage:
#   bash render.sh <scene_name> [quality]
#   bash render.sh all [quality]
#
#   <scene_name> : basename of scenes/<scene_name>.py (e.g. 01_tokenization)
#   [quality]    : "low" (480p15, default, smoke) | "high" (1080p30, final)
#   all          : render all 7 scenes sequentially
#
# The scene class is discovered automatically (first `class <Name>(...Scene...)`
# match in the file), so callers never need to pass the class name.

set -euo pipefail

# All 7 scenes in narrative order (matches script.md flow).
ALL_SCENES=(
  "01_tokenization"
  "02_llm_numbers"
  "03_image_patches"
  "04_audio_chunks"
  "05_shared_vector_space"
  "06_two_path_comparison"
  "07_full_pipeline"
)

SCENE_NAME="${1:-}"
QUALITY="${2:-low}"

if [[ -z "${SCENE_NAME}" ]]; then
  echo "Usage: bash render.sh <scene_name> [low|high]" >&2
  echo "       bash render.sh all [low|high]" >&2
  exit 2
fi

# --- "all" case: recursively re-invoke ourselves per scene -----------------
if [[ "${SCENE_NAME}" == "all" ]]; then
  echo "╭─ render.sh all ──────────────────────────────────────────────╮"
  echo "│ Rendering all ${#ALL_SCENES[@]} scenes at quality=${QUALITY}       │"
  echo "╰─────────────────────────────────────────────────────────────╯"
  failed=()
  for scene in "${ALL_SCENES[@]}"; do
    echo ""
    echo "━━━ ${scene} ━━━"
    if ! bash "$0" "${scene}" "${QUALITY}"; then
      failed+=("${scene}")
    fi
  done
  echo ""
  if (( ${#failed[@]} > 0 )); then
    echo "✗ ${#failed[@]} scene(s) failed: ${failed[*]}"
    exit 6
  fi
  echo "✓ All ${#ALL_SCENES[@]} scenes rendered at ${QUALITY}."
  echo "Run \`bash verify.sh\` to confirm all 7 MP4s are present."
  exit 0
fi

# --- single-scene case --------------------------------------------------
SCENE_FILE="scenes/${SCENE_NAME}.py"
if [[ ! -f "${SCENE_FILE}" ]]; then
  echo "Scene file not found: ${SCENE_FILE}" >&2
  exit 3
fi

# Discover the Scene class name (first match: `class <Name>(...Scene...)`).
# Accepts both `class X(Scene)` and `class X(ThreeDScene)` / `class X(SomeScene)`.
# Use awk (portable across GNU/BSD) rather than `grep -oP` (GNU-only).
SCENE_CLASS=$(awk '
  /^[ \t]*class[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*\)/ {
    match($0, /class[ \t]+[A-Za-z_][A-Za-z0-9_]*/)
    name = substr($0, RSTART, RLENGTH)
    sub(/^class[ \t]+/, "", name)
    print name
    exit
  }
' "${SCENE_FILE}")

if [[ -z "${SCENE_CLASS}" ]]; then
  echo "No \`class X(Scene)\` definition found in ${SCENE_FILE}" >&2
  exit 4
fi

mkdir -p output

case "${QUALITY}" in
  low)  MANIM_FLAGS=(-ql) ;;                 # 480p15 — smoke
  high) MANIM_FLAGS=(-qh --fps 30) ;;        # 1080p30 — final (script NFR)
  *)
    echo "Unknown quality '${QUALITY}'. Use 'low' (default) or 'high'." >&2
    exit 5
    ;;
esac

echo "→ Rendering ${SCENE_NAME} :: ${SCENE_CLASS}  (${QUALITY})"
# --media_dir output/ makes the AC's "output/01_tokenization.mp4" contract literal:
# Manim writes to media/videos/<name>/<quality>/<Class>.mp4 by default, which
# we redirect to output/. The mp4 is also renamed from <Class>.mp4 to
# <scene_name>.mp4 by the post-render block below.
exec manim \
  --media_dir output \
  --output_file "${SCENE_NAME}" \
  "${MANIM_FLAGS[@]}" --format mp4 \
  "${SCENE_FILE}" "${SCENE_CLASS}"
