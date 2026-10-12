# Régénère notebooks/transcription_colab.ipynb : python tools/make_colab_notebook.py notebooks/transcription_colab.ipynb
import json, sys

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
def code(src, title=None):
    meta = {"cellView": "form"} if title else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}

cells = [
md("""
# 🎙️ Transcription audio sur GPU gratuit (Google Colab)

Utilisez ce carnet si l'ordinateur est trop lent pour transcrire dans le navigateur, par exemple sur un vieux Mac ou pour un enregistrement de plusieurs heures.
Le calcul se fait sur une carte graphique de Google, gratuitement. Comptez environ **15 à 25 minutes pour 3 h d'audio**.

**Mode d'emploi :**
1. Connectez-vous avec un compte Google.
2. Menu **Exécution → Modifier le type d'exécution → GPU T4**, puis **Enregistrer**.
3. Menu **Exécution → Tout exécuter**. Si Colab demande une confirmation, choisissez « Exécuter quand même ».
4. Un bouton **« Choisir les fichiers »** apparaît sous l'étape 2 : sélectionnez vos fichiers audio ou vidéo.
5. À la fin, un fichier `transcriptions.zip` se télécharge automatiquement. Il contient le texte (`.txt`) et les sous-titres (`.srt`, `.vtt`).

Pour les gros fichiers (plus de 500 Mo), il vaut mieux les mettre dans Google Drive et remplir `dossier_drive` à l'étape 2.
"""),
code("""
#@title 1. Installation (environ 1 minute)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/quentin91590/Translate.git@claude/admiring-cray-dmxzoy"], check=True)

import torch
if torch.cuda.is_available():
    print("✅ GPU détecté :", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "❌ Pas de carte graphique (GPU). Menu Exécution → Modifier le type d'exécution → GPU T4 → Enregistrer, "
        "puis Exécution → Tout exécuter. Si l'option est déjà choisie, Google n'a pas de GPU libre "
        "(quota gratuit atteint ou forte affluence) : réessayez plus tard, ou utilisez la version dans le navigateur."
    )
""", title=True),
code("""
#@title 2. Choisir les fichiers et transcrire
langue = "auto"  #@param ["auto", "fr", "en", "es", "de", "it", "pt", "nl", "ar", "zh", "ja", "ko", "ru", "tr", "pl"]
plusieurs_langues = False  #@param {type:"boolean"}
modele = "large-v3"  #@param ["large-v3", "large-v3-turbo"]
#@markdown Optionnel : dossier Google Drive contenant les fichiers (ex. `MyDrive/audios`). Laissez vide pour envoyer les fichiers depuis l'ordinateur.
dossier_drive = ""  #@param {type:"string"}

import os, shutil, time
from pathlib import Path

os.environ["WHISPER_MODEL"] = modele
from transcriber.backends import get_backend
from transcriber.config import get_settings
from transcriber import pipeline

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".mp4", ".mov", ".mkv", ".webm", ".wma", ".aiff"}
entrees = Path("/content/entrees"); entrees.mkdir(exist_ok=True)
sorties = Path("/content/transcriptions"); shutil.rmtree(sorties, ignore_errors=True); sorties.mkdir()

if dossier_drive.strip():
    from google.colab import drive
    drive.mount("/content/drive")
    # Accepte « audios », « Mon Drive/audios », « MyDrive/audios » ou un chemin copié depuis Colab.
    rel = dossier_drive.strip().replace(chr(92), "/")  # barres obliques inverses -> /
    for prefixe in ("/content/drive/", "content/drive/"):
        if rel.startswith(prefixe):
            rel = rel[len(prefixe):]
    rel = rel.strip("/")
    for alias in ("Mon Drive", "My Drive", "Mon drive"):
        if rel == alias or rel.startswith(alias + "/"):
            rel = "MyDrive" + rel[len(alias):]
    if not (rel.startswith("MyDrive") or rel.startswith("Shareddrives")):
        rel = "MyDrive/" + rel
    source = Path("/content/drive") / rel
    if not source.is_dir():
        dossiers = sorted(p.name for p in Path("/content/drive/MyDrive").iterdir() if p.is_dir())
        raise SystemExit(f"❌ Dossier introuvable : {rel}. Dossiers présents dans « Mon Drive » : {', '.join(dossiers[:30])}")
    fichiers = sorted(p for p in source.iterdir() if p.suffix.lower() in AUDIO_EXT)
else:
    from google.colab import files
    envoyes = files.upload()
    fichiers = []
    for nom, contenu in envoyes.items():
        p = entrees / nom
        p.write_bytes(contenu)
        fichiers.append(p)

if not fichiers:
    raise SystemExit("Aucun fichier audio trouvé.")

print("Chargement du modèle", modele, "…")
backend = get_backend("whisper", get_settings())

for f in fichiers:
    print(f"\\n▶ {f.name}")
    t0 = time.time()
    derniere = [-1]
    def progression(frac, msg):
        pct = int(frac * 100)
        if pct >= derniere[0] + 5:
            derniere[0] = pct
            print(f"  {pct:3d} %  {msg or ''}")
    dossier = sorties / f.stem
    t = pipeline.run(f, dossier, backend, None if langue == "auto" else langue,
                     plusieurs_langues, None, progression)
    for ext in ("txt", "srt", "vtt", "json"):
        (dossier / f"transcript.{ext}").rename(dossier / f"{f.stem}.{ext}")
    print(f"  ✅ langue : {t.language} · {t.duration/60:.0f} min d'audio · {(time.time()-t0)/60:.1f} min de calcul")

archive = shutil.make_archive("/content/transcriptions", "zip", sorties)
from google.colab import files as _f
_f.download(archive)
print("\\n📦 Téléchargement de transcriptions.zip lancé.")
""", title=True),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
