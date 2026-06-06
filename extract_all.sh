#!/usr/bin/env bash
#
# extract_all.sh — run all 4 extraction scripts in dependency order.
#
# Order matters:
#   1. extract_text          (CLIP tokenizer → tokens.npy)
#   2. extract_image_patches (ViT-16 → patch_embeddings.npy)
#   3. extract_audio         (Whisper + librosa → audio_*.npy)
#   4. extract_shared_space  (CLIPModel → pca_coords_3d.npy)
#
# The first three have no inter-dependencies and could run in parallel.
# extract_shared_space depends on CLIPModel (already loaded by text+image
# scripts in the same process? No — each is a separate process), so we
# run it last to avoid the 600MB model load happening twice.
#
# Usage: bash extract_all.sh
#
# Note: HuggingFace downloads happen on first run (~1-2 GB across all
# three models). Subsequent runs use the local cache.

set -euo pipefail

cd "$(dirname "$0")"

echo "╭─ extract_all.sh ────────────────────────────────────────────╮"
echo "│ Running all 4 extraction scripts in dependency order.       │"
echo "╰─────────────────────────────────────────────────────────────╯"

run_step() {
  local step_name="$1"
  local script="$2"
  echo ""
  echo "→ ${step_name}: python ${script}"
  python "${script}"
  echo "✓ ${step_name} done"
}

run_step "1/4 Text tokens"      extract/extract_text.py
run_step "2/4 Image patches"   extract/extract_image_patches.py
run_step "3/4 Audio mel+wave"  extract/extract_audio.py
run_step "4/4 Shared space"    extract/extract_shared_space.py

echo ""
echo "✓ All extractions complete. Outputs in data/."
