# Transcripteur audio

Application web et en ligne de commande pour **transcrire de longs enregistrements audio ou vidéo (jusqu'à ~3 h)** en texte, avec **détection automatique de la langue**, en utilisant uniquement des **modèles open source qui tournent sur votre machine**. Aucun fichier n'est envoyé à un service externe.

## Modèles

| Moteur | Modèle | Licence | Langues | Points forts |
|---|---|---|---|---|
| `whisper` (par défaut) | Whisper **large-v3** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | 99 | Couvre le plus de langues, très robuste sur l'audio long, horodatages précis, tourne aussi sur CPU |
| `qwen` (optionnel) | [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 | 52 langues et dialectes | Souvent plus précis que Whisper sur les langues qu'il couvre ; langue détectée morceau par morceau. GPU recommandé |

Pourquoi ce choix par défaut : en octobre 2026, les modèles en tête du [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (NVIDIA Canary-Qwen, Parakeet…) sont surtout anglais ou limités à environ 25 langues européennes. Pour des fichiers dont la langue est inconnue, Whisper large-v3 reste le choix le plus sûr, et Qwen3-ASR est la meilleure alternative multilingue récente. Le moteur se choisit à chaque transcription dans l'interface.

## Fonctionnalités

- Interface web : glisser-déposer, file d'attente, barre de progression avec temps restant estimé, annulation, aperçu du texte.
- Formats d'entrée : tout ce que lit ffmpeg (mp3, wav, m4a, aac, ogg, opus, flac, mp4, mkv, webm…).
- Exports : **TXT**, **SRT** et **VTT** (sous-titres) et **JSON** (segments, langue, horodatages).
- Détection de la langue (ou langue imposée), avec une option pour les enregistrements multilingues.
- Audio long :
  - Whisper parcourt le fichier avec une fenêtre glissante. Le filtre VAD Silero saute les silences, et le texte n'est pas conditionné sur la fenêtre précédente, ce qui évite les boucles de répétition sur plusieurs heures.
  - Qwen3-ASR traite des morceaux d'environ 4 minutes, coupés dans des silences pour ne jamais couper un mot.
- Les tâches sont enregistrées sur disque (`data/jobs/`) et reprennent si le serveur redémarre.

## Installation

Prérequis : Python ≥ 3.10 et **ffmpeg**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optionnel, pour le moteur Qwen3-ASR (GPU NVIDIA) :
pip install -r requirements-qwen.txt
```

Au premier lancement, le modèle est téléchargé depuis Hugging Face (environ 3 Go pour Whisper large-v3, 4 Go pour Qwen3-ASR-1.7B).

## Utilisation

### Interface web

```bash
python -m transcriber            # http://127.0.0.1:8000
python -m transcriber --host 0.0.0.0 --port 8000   # accessible depuis le réseau
```

### Ligne de commande

```bash
python -m transcriber.cli interview.mp3                  # langue détectée automatiquement
python -m transcriber.cli conference.mp4 -l fr -o sorties/
python -m transcriber.cli podcast.m4a -b qwen --multilingual
```

### Docker (GPU)

```bash
docker build -t transcriber .                    # ajoutez --build-arg WITH_QWEN=1 pour Qwen3-ASR
docker run --gpus all -p 8000:8000 -v $PWD/data:/data -v $PWD/models:/models transcriber
```

## Configuration (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `TRANSCRIBER_BACKEND` | `whisper` | Moteur par défaut |
| `WHISPER_MODEL` | `large-v3` | `large-v3-turbo` est environ 6× plus rapide pour une précision un peu moindre |
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ou `Qwen/Qwen3-ASR-0.6B` (plus léger) |
| `TRANSCRIBER_DEVICE` | `auto` | `cuda` ou `cpu` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `float16` sur GPU, `int8` sur CPU |
| `TRANSCRIBER_MAX_DURATION` | `12600` | Durée maximale acceptée, en secondes (3 h 30) |
| `TRANSCRIBER_CHUNK_SECONDS` | `240` | Taille des morceaux pour Qwen3-ASR |
| `TRANSCRIBER_DATA_DIR` | `data` | Stockage des fichiers et des résultats |

## Performances (ordre de grandeur pour 3 h d'audio)

- GPU récent (RTX 4090 / A100), Whisper large-v3 en mode batch : environ 5 à 10 minutes.
- GPU avec 8 Go de VRAM : environ 15 à 30 minutes.
- CPU seul : plusieurs heures avec large-v3. Utilisez plutôt `WHISPER_MODEL=large-v3-turbo` ou `medium`.
- Mémoire vive : prévoyez environ 2 Go de plus que le modèle, car le signal de 3 h est décodé en mémoire.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## API HTTP

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/jobs` | multipart : `file`, `backend`, `language`, `multilingual` |
| `GET` | `/api/jobs` / `/api/jobs/{id}` | État et progression |
| `POST` | `/api/jobs/{id}/cancel` | Annuler |
| `DELETE` | `/api/jobs/{id}` | Supprimer |
| `GET` | `/api/jobs/{id}/transcript.{txt,srt,vtt,json}` | Télécharger le résultat |
