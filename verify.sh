#!/usr/bin/env bash
#
# verify.sh — check that all 7 expected MP4 files exist and are non-empty.
#
# The 7 expected outputs (one per scene) live under
#   output/videos/<scene_name>/<quality>/<scene_name>.mp4
# where quality is "480p15" (smoke) or "1080p30" (high).
#
# We check the smoke quality (480p15) by default; pass QUALITY=high to
# check the 1080p30 outputs instead.
#
# Usage:  bash verify.sh
#         QUALITY=high bash verify.sh
# Exit:   0 if all 7 outputs are present + non-empty; 1 otherwise.

set -euo pipefail

cd "$(dirname "$0")"

QUALITY="${QUALITY:-480p15}"
EXPECTED_DIR="output/videos"

EXPECTED_SCENES=(
  "01_tokenization"
  "02_llm_numbers"
  "03_image_patches"
  "04_audio_chunks"
  "05_shared_vector_space"
  "06_two_path_comparison"
  "07_full_pipeline"
)

MISSING=()
EMPTY=()

for scene in "${EXPECTED_SCENES[@]}"; do
  mp4="${EXPECTED_DIR}/${scene}/${QUALITY}/${scene}.mp4"
  if [[ ! -f "${mp4}" ]]; then
    MISSING+=("${mp4}")
  elif [[ ! -s "${mp4}" ]]; then
    EMPTY+=("${mp4}")
  fi
done

if (( ${#MISSING[@]} > 0 )); then
  echo "✗ Missing ${#MISSING[@]} MP4 file(s):"
  for m in "${MISSING[@]}"; do
    echo "    - ${m}"
  done
fi

if (( ${#EMPTY[@]} > 0 )); then
  echo "✗ Empty ${#EMPTY[@]} MP4 file(s):"
  for e in "${EMPTY[@]}"; do
    echo "    - ${e}"
  done
fi

if (( ${#MISSING[@]} > 0 )) || (( ${#EMPTY[@]} > 0 )); then
  echo ""
  echo "Run \`bash render.sh all\` first to produce the MP4s."
  exit 1
fi

echo "✓ All 7 expected MP4s present at ${QUALITY} (${EXPECTED_DIR}/)."
