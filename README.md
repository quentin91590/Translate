# Transcripteur audio

Application pour **transcrire de longs enregistrements audio ou vidéo (jusqu'à ~3 h)** en texte, avec **détection automatique de la langue**, en utilisant uniquement des **modèles open source qui tournent sur votre machine**. Aucun fichier n'est envoyé à un service externe.

Trois façons de l'utiliser :

| | Pour qui | Où tourne le calcul |
|---|---|---|
| **Page web** (`docs/`, publiée sur GitHub Pages) | N'importe qui, sans rien installer : idéal sur un Mac avec puce Apple | Dans le navigateur, sur la carte graphique du Mac |
| **Google Colab** (`notebooks/transcription_colab.ipynb`) | Mac trop lent ou ancien Mac Intel | Sur un GPU gratuit de Google |
| **Serveur Python** (`transcriber/`) | Un PC avec une carte NVIDIA, ou un serveur | Sur la machine qui héberge le serveur |

## Page web (GitHub Pages)

La page `docs/index.html` fait tourner **Whisper directement dans le navigateur** grâce à [Transformers.js](https://github.com/huggingface/transformers.js) et WebGPU. Rien n'est installé et aucun fichier audio ne quitte l'ordinateur. Le modèle est téléchargé une seule fois depuis Hugging Face, puis gardé en cache.

### Mise en ligne (une seule fois)

1. Sur GitHub : **Settings → Pages**.
2. **Source** : « Deploy from a branch ». **Branch** : `claude/admiring-cray-dmxzoy`, dossier **`/docs`**, puis **Save**.
3. Après 1 à 2 minutes, la page est en ligne sur **https://quentin91590.github.io/Translate/**.

Si le dossier `/ (root)` est choisi à la place de `/docs`, la racine redirige automatiquement vers `docs/`.

### Ce qu'il faut savoir côté Mac

| Mac | Navigateur | Mode conseillé | 3 h d'audio (ordre de grandeur) |
|---|---|---|---|
| Puce Apple (M1 à M4), 16 Go de RAM ou plus | **Google Chrome** | Haute précision (Whisper large-v3 turbo) | de 30 min (puces récentes Pro/Max) à 2–3 h (M1) |
| Puce Apple, 8 Go de RAM | Google Chrome | Équilibré (Whisper small) | 30 min à 1 h |
| Mac Intel | Google Chrome | Équilibré, ou **Google Colab** | plusieurs heures dans le navigateur, 15 à 25 min sur Colab |

- **Chrome plutôt que Safari** : Safari n'utilise la carte graphique qu'à partir de macOS Tahoe (26), et il recharge parfois les pages qui consomment beaucoup de mémoire.
- **Téléchargement du modèle** : environ 1,6 Go pour « Haute précision », 600 Mo pour « Équilibré », 200 Mo pour « Rapide », la première fois seulement.
- **Pendant la transcription**, garder l'onglet affiché et le Mac branché. La page empêche la mise en veille de l'écran.
- **Si la page se ferme**, il suffit de rechoisir le même fichier : la transcription reprend où elle s'était arrêtée.

### Fonctionnement technique

- **Décodage en flux** : le fichier est lu avec [Mediabunny](https://mediabunny.dev) (WebCodecs) et rééchantillonné à la volée en 16 kHz mono. La mémoire reste constante ; décoder 3 h d'un coup demanderait plusieurs Go.
- **Fenêtres de 30 s** avec la méthode « seek » d'OpenAI : une phrase coupée en fin de fenêtre est reprise au début de la suivante.
- **Silence et hallucinations** : les fenêtres silencieuses sont sautées, et les génériques inventés par Whisper (« Sous-titres réalisés par… ») ainsi que les boucles de répétition sont filtrés.
- **Détection de la langue** : Transformers.js 4.3.1 ne la fait pas (sans langue, il transcrit en anglais). La page interroge donc le modèle sur les jetons de langue, à partir des 3 passages les plus parlés des 5 premières minutes.
- **Fuite de mémoire GPU** de Transformers.js 4.3.1 avec Whisper ([transformers.js#1739](https://github.com/huggingface/transformers.js/issues/1739)) : elle est corrigée à l'exécution dans `docs/asr-worker.js`. Par sécurité, le moteur est aussi recréé toutes les 30 min d'audio.
- **Sauvegarde** : la progression est enregistrée dans IndexedDB après chaque fenêtre. Les transcriptions terminées restent disponibles dans la page.

Ces durées sont des estimations tirées de mesures publiées. La page affiche le vrai temps restant après quelques minutes et propose un mode plus léger si le Mac est trop lent.

### Tests de la page web

```bash
node --test tests/web/core.test.mjs          # logique pure (fenêtres, filtres, rééchantillonnage, formats)
# Bout en bout dans Chromium avec un vrai modèle (fichiers servis localement) :
MODEL_DIR=… ORT_DIST=… TJS_DIST=… AUDIO=reunion.mp3 node tests/web/e2e.mjs
```

Paramètres d'URL de diagnostic : `?debug=1` affiche le temps par fenêtre et la mémoire GPU, `?device=wasm` force le processeur, `?model=<id>&dtype=<q8|fp32…>` change de modèle.

---

## Serveur Python

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
