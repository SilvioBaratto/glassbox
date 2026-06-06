# glassbox

Visualizzatore multimodale: mostriamo, fotogramma per fotogramma, come un modello AI
"legge" testo, immagini e audio. Seguiamo la narrazione di `script.md` e la trasformiamo
in scene animate con [Manim Community](https://www.manim.community/).

> **Architettura: extract-first, animate-second.** I modelli HuggingFace
> (`openai/clip-vit-base-patch32`, `google/vit-base-patch16-224`, `openai/whisper-base`)
> vengono eseguiti **una volta offline** in `extract/`. Le scene Manim leggono solo
> array `.npy` già salvati in `data/` — niente inferenza dentro il render loop.

## Installazione

```bash
# 1. ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# 2. dipendenze base
python -m pip install -r requirements.txt

# 3. PyTorch (wheel separata per piattaforma — scegli la tua)
# macOS (Apple Silicon o Intel):
python -m pip install torch torchvision torchaudio
# Linux + CUDA 12.1:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Linux + CPU only:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Manim e dipendenze di sistema (vedi Troubleshooting)
python -m pip install manim==0.18.1
# macOS: brew install cairo pkg-config
```

## Flusso di lavoro (end-to-end)

Gli script `extract_all.sh`, `render.sh all` e `verify.sh` formano la pipeline
completa. Ognuno è idempotente.

### 1. Estrai i tensori (una volta, offline)

```bash
bash extract_all.sh
```

Questo esegue i 4 script di estrazione in ordine di dipendenza:
1. `extract/extract_text.py`          — CLIP tokenizer
2. `extract/extract_image_patches.py` — ViT-16
3. `extract/extract_audio.py`         — Whisper + librosa
4. `extract/extract_shared_space.py`  — CLIPModel + PCA

**Tempo**: ~1-2 minuti la prima volta (download modelli ~1-2 GB), ~30s le successive
(modelli in cache locale).

**Output**: file `.npy` in `data/` (ignorati da Git).

### 2. Renderizza tutte le 7 scene

```bash
bash render.sh all
```

Oppure una singola scena:
```bash
bash render.sh 01_tokenization       # smoke: 480p15
bash render.sh 07_full_pipeline high # finale: 1080p30
```

Le 7 scene in ordine narrativo:
| #  | Scene                  | Durata (~) | Colore  |
|----|------------------------|------------|---------|
| 01 | `01_tokenization`     | 5s         | blu     |
| 02 | `02_llm_numbers`      | 8s         | blu     |
| 03 | `03_image_patches`    | 10s        | arancione|
| 04 | `04_audio_chunks`     | 10s        | verde   |
| 05 | `05_shared_vector_space` (3D) | 8s  | blu+arancione |
| 06 | `06_two_path_comparison` | 12s      | mix     |
| 07 | `07_full_pipeline`    | 12s        | mix     |

**Tempo totale**: ~2-3 minuti a 480p15 (smoke), ~5-10 minuti a 1080p30.

**Output**: 7 file MP4 in `output/videos/<scene_name>/480p15/<scene_name>.mp4`.

### 3. Verifica che tutti gli output ci siano

```bash
bash verify.sh
```

Esce 0 se tutti i 7 MP4 sono presenti e non vuoti, 1 altrimenti (con elenco
dei file mancanti/vuoti).

### Pipeline completa in un comando

```bash
bash extract_all.sh && bash render.sh all && bash verify.sh
```

## Struttura del progetto

```
extract/   ← script Python (girano una volta, salvano .npy)
data/      ← tensori estratti (gitignored)
scenes/    ← scene Manim, leggono da data/
assets/    ← immagini e audio di esempio
script.md  ← fonte di verità della narrazione (italiano)
extract_all.sh  ← esegue tutti gli script di estrazione
render.sh  ← renderizza una scena (o `all`)
verify.sh  ← conferma che i 7 MP4 siano presenti
tests/     ← test strutturali (pytest)
```

## Test

```bash
python -m pytest tests/
```

150+ test strutturali che verificano:
- Layout del progetto, presenza file, `.gitignore`
- Asset sample (image + audio)
- 4 script di estrazione (mockano HuggingFace per essere ermetici)
- 7 scene Manim (mockano mobject per evitare il render loop)
- Orchestratore (extract_all.sh, render.sh all, verify.sh, README)

I test **non** validano l'output visivo — quello lo fa un'ispezione manuale
delle 7 scene renderizzate.

## Troubleshooting

### `ModuleNotFoundError: No module named 'librosa'`

`librosa` è in `requirements.txt` ma potrebbe non essere installato nel
tuo ambiente. Soluzione:
```bash
python -m pip install librosa soundfile
```

### `manim: command not found` / errori di import

Manim richiede dipendenze di sistema Cairo e pkg-config.

**macOS**:
```bash
brew install cairo pkg-config
PKG_CONFIG_PATH=/opt/homebrew/opt/cairo/lib/pkgconfig python -m pip install manim==0.18.1
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install libcairo2-dev libpango1.0-dev ffmpeg
python -m pip install manim==0.18.1
```

### `FileNotFoundError: ffmpeg`

Manim richiede `ffmpeg` per l'encoding MP4.

**macOS**: `brew install ffmpeg`
**Linux**: `sudo apt-get install ffmpeg`

### `HTTPSConnectionPool ... Read timed out` durante l'estrazione

Prima esecuzione: HuggingFace scarica i 3 modelli (~1-2 GB totali). Riavvia
`bash extract_all.sh` per riprendere da dove si era interrotto (i modelli
parzialmente scaricati sono in `~/.cache/huggingface/`).

### Output mancanti dopo `bash render.sh all`

```bash
bash verify.sh
```

Stampa l'elenco dei file mancanti. Quindi rilancia `bash render.sh <scene>`
solo per le scene mancanti.

### Test lenti

I test sono intenzionalmente veloci (~3s totali) — mockano tutti i
modelli HuggingFace. Se noti rallentamenti, controlla che nessun test
stia importando `transformers` o `whisper` direttamente.

## Licenza

MIT — vedi `LICENSE`.
