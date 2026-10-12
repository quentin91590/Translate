// Chef d'orchestre de la page : lecture du fichier, chargement du modèle,
// détection de la langue, boucle de transcription, sauvegarde et exports.
import {
  SR, WINDOW_S, mergeWindow, isHallucination, isRepeatLoop, isSilent, pickLanguageWindows,
  combineLanguageVotes, formatDuration, PcmBuffer,
} from "./core.js";
import { FORMATS, MIME } from "./formats.js";

const params = new URLSearchParams(location.search);
const DEBUG = params.has("debug");
const RECYCLE_WINDOWS = Number(params.get("recycle")) || 60; // ~30 min d'audio par worker
// Délai max. d'une fenêtre avant de considérer le moteur bloqué : large au début,
// puis 8 fois la durée moyenne observée (jamais moins de 5 min).
const windowTimeout = (avgMs) => (avgMs ? Math.max(5 * 60e3, 8 * avgMs) : 30 * 60e3);
const COLAB_URL =
  "https://colab.research.google.com/github/quentin91590/Translate/blob/claude/admiring-cray-dmxzoy/notebooks/transcription_colab.ipynb";

const MODELS = {
  turbo: { id: "onnx-community/whisper-large-v3-turbo", name: "Haute précision", detail: "Whisper large-v3 turbo", gpuOnly: true },
  small: { id: "onnx-community/whisper-small", name: "Équilibré", detail: "Whisper small" },
  base: { id: "onnx-community/whisper-base", name: "Rapide", detail: "Whisper base" },
};

const $ = (id) => document.getElementById(id);
const langName = (() => {
  let dn = null;
  try { dn = new Intl.DisplayNames(["fr"], { type: "language" }); } catch {}
  return (code) => {
    if (!code) return "inconnue";
    let n = code;
    try { n = dn?.of(code === "jw" ? "jv" : code) || code; } catch {}
    return n.charAt(0).toUpperCase() + n.slice(1);
  };
})();

const state = {
  caps: null,
  file: null,
  info: null, // { path: 'stream'|'whole', duration, codec }
  running: false,
  cancel: false,
  asr: null, // moteur en cours (pour pouvoir l'arrêter immédiatement)
  result: null,
};

// ---------------------------------------------------------------- workers

class WorkerClient {
  constructor(url, onEvent) {
    this.worker = new Worker(new URL(url, import.meta.url), { type: "module" });
    this.seq = 0;
    this.pending = new Map();
    this.worker.onmessage = ({ data }) => {
      if (data.id == null) return onEvent?.(data);
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      clearTimeout(p.timer);
      data.ok ? p.resolve(data.result) : p.reject(new Error(data.error));
    };
    this.worker.onerror = (e) => this.failAll(new Error(e.message || "Le moteur s'est arrêté."));
  }
  call(type, payload = {}, transfer = [], timeoutMs = 0) {
    if (this.dead) return Promise.reject(new Error("CANCELLED"));
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      const timer = timeoutMs ? setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("TIMEOUT"));
      }, timeoutMs) : null;
      this.pending.set(id, { resolve, reject, timer });
      this.worker.postMessage({ id, type, ...payload }, transfer);
    });
  }
  failAll(err) {
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(err); }
    this.pending.clear();
  }
  terminate() {
    this.dead = true;
    this.worker.terminate();
    this.failAll(new Error("CANCELLED"));
  }
}

// ---------------------------------------------------------------- sources audio

/** Lecture en flux par le worker de décodage (WebCodecs). Mémoire constante. */
class StreamSource {
  constructor(file) { this.file = file; this.client = new WorkerClient("./decoder-worker.js"); }
  open(startSec) { return this.client.call("open", { file: this.file, startSec }); }
  pull(seconds) { return this.client.call("pull", { seconds }); }
  close() { this.client.terminate(); }
}

/** Repli : décodage complet par le navigateur (anciens Safari, AIFF/CAF/ALAC). */
class WholeSource {
  static async create(file) {
    const ctx = new OfflineAudioContext(1, 1, SR);
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const n = buf.length, ch = buf.numberOfChannels;
    const mono = buf.getChannelData(0); // moyenne faite en place : pas de copie de plusieurs centaines de Mo
    for (let c = 1; c < ch; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += d[i];
    }
    if (ch > 1) for (let i = 0; i < n; i++) mono[i] /= ch;
    const s = new WholeSource();
    s.data = mono;
    s.pos = 0;
    return s;
  }
  get duration() { return this.data.length / SR; }
  async open(startSec) { this.pos = Math.round(startSec * SR); }
  async pull(seconds) {
    const end = Math.min(this.data.length, this.pos + Math.round(seconds * SR));
    const pcm = this.data.slice(this.pos, end);
    this.pos = end;
    return { pcm, eof: end >= this.data.length };
  }
  close() { this.data = null; }
}

// ---------------------------------------------------------------- sauvegarde (IndexedDB)

const store = {
  db: null,
  async open() {
    if (this.db) return this.db;
    this.db = await new Promise((resolve, reject) => {
      const req = indexedDB.open("transcripteur", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("jobs", { keyPath: "key" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    this.db.onclose = () => { this.db = null; };
    this.db.onversionchange = () => { this.db?.close(); this.db = null; };
    return this.db;
  },
  async tx(mode, fn) {
    try {
      const db = await this.open();
      return await new Promise((resolve, reject) => {
        const t = db.transaction("jobs", mode);
        const r = fn(t.objectStore("jobs"));
        t.oncomplete = () => resolve(r?.result);
        t.onerror = () => reject(t.error);
        t.onabort = () => reject(t.error || new DOMException("Transaction annulée", "AbortError"));
      });
    } catch (e) {
      console.warn("Sauvegarde indisponible", e);
      this.db = null;
      this.failed = true;
      return undefined;
    }
  },
  get(key) { return this.tx("readonly", (s) => s.get(key)); },
  put(rec) { return this.tx("readwrite", (s) => s.put(rec)); },
  del(key) { return this.tx("readwrite", (s) => s.delete(key)); },
  all() { return this.tx("readonly", (s) => s.getAll()); },
};

const fileKey = (f) => `${f.name}|${f.size}|${f.lastModified}`;

// ---------------------------------------------------------------- capacités

async function detectCapabilities() {
  const ua = navigator.userAgent;
  const caps = {
    webgpu: false, f16: false, vendor: "",
    webcodecs: typeof AudioDecoder !== "undefined",
    safari: /Safari/.test(ua) && !/Chrome|Chromium|Edg|OPR|Firefox/.test(ua),
    chrome: /Chrome|Chromium/.test(ua),
    mac: /Macintosh|Mac OS X/.test(ua),
  };
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter) {
      caps.webgpu = true;
      caps.f16 = adapter.features.has("shader-f16");
      caps.vendor = (adapter.info?.vendor || "").toLowerCase();
    }
  } catch {}
  caps.appleGpu = caps.vendor.includes("apple");
  return caps;
}

function renderCapabilities(c) {
  const lines = [];
  if (c.webgpu && c.appleGpu) {
    lines.push(["ok", "Votre Mac peut utiliser sa puce graphique Apple : la transcription sera rapide."]);
    if (c.safari) lines.push(["info", "Avec Safari, les longs fichiers en « Haute précision » peuvent faire recharger la page. Google Chrome est plus sûr."]);
  } else if (c.webgpu) {
    lines.push(["ok", "La carte graphique est utilisable (WebGPU)."]);
    if (c.mac) lines.push(["info", "Sur un Mac Intel, le mode « Équilibré » est conseillé. Pour de très longs fichiers, le mode Google Colab (plus bas) sera plus rapide."]);
  } else {
    lines.push(["warn", c.safari
      ? "Safari ne donne pas accès à la carte graphique sur ce Mac (il faut macOS Tahoe). Ouvrez cette page dans Google Chrome pour aller beaucoup plus vite."
      : "Ce navigateur ne donne pas accès à la carte graphique : la transcription utilisera le processeur et sera lente. Essayez Google Chrome à jour."]);
  }
  if (!c.webcodecs) {
    lines.push(["warn", "Ce navigateur est ancien : les très longs fichiers risquent de saturer la mémoire. Google Chrome est recommandé."]);
  }
  $("caps").innerHTML = lines.map(([k, t]) => `<li class="${k}">${t}</li>`).join("");
}

function recommendedModel(c) {
  // Safari recharge les pages trop gourmandes en mémoire : modèle plus léger.
  if (c.webgpu && c.appleGpu && !c.safari) return "turbo";
  if (c.webgpu && c.appleGpu) return "small";
  if (c.webgpu) return "small";
  return "base";
}

function modelSpec(key, { forceCpu = false } = {}) {
  const device = state.caps.webgpu && !forceCpu ? "webgpu" : "wasm";
  if (MODELS[key].gpuOnly && device === "wasm") key = "small";
  const m = MODELS[key];
  let dtype;
  if (device === "wasm") dtype = "q8";
  else if (key === "turbo") dtype = { encoder_model: state.caps.f16 ? "fp16" : "q4", decoder_model_merged: "q4" };
  else dtype = { encoder_model: "fp32", decoder_model_merged: "q4" };
  // Surcharges de diagnostic : ?model=<id huggingface>&dtype=<q8|fp32|...>
  const id = params.get("model") || m.id;
  if (params.get("dtype")) dtype = params.get("dtype");
  if (params.get("device")) return { key, id, dtype, device: params.get("device") };
  return { key, id, device, dtype };
}

const SIZES = { // Mo téléchargés une seule fois (puis gardés en cache)
  turbo: { fp16: 1600, q4: 760, wasm: 0 },
  small: { webgpu: 590, wasm: 250 },
  base: { webgpu: 210, wasm: 80 },
};

function renderModelChoices() {
  const rec = recommendedModel(state.caps);
  const items = Object.entries(MODELS).map(([key, m]) => {
    const disabled = m.gpuOnly && !state.caps.webgpu;
    const size = key === "turbo" ? SIZES.turbo[state.caps.f16 ? "fp16" : "q4"] : SIZES[key][state.caps.webgpu ? "webgpu" : "wasm"];
    const desc = {
      turbo: "La meilleure qualité. Pour Mac avec puce Apple et 16 Go de mémoire ou plus (menu Pomme → À propos de ce Mac). Avec 8 Go, choisissez « Équilibré ».",
      small: "Bon compromis vitesse / qualité. Conseillé avec 8 Go de mémoire ou sur les Mac plus anciens.",
      base: "Le plus rapide, mais plus d'erreurs. À réserver aux ordinateurs lents.",
    }[key];
    return `<label class="choice${disabled ? " disabled" : ""}">
      <input type="radio" name="model" value="${key}" ${key === rec ? "checked" : ""} ${disabled ? "disabled" : ""}>
      <span><strong>${m.name}</strong>${key === rec ? ' <em class="tag">conseillé</em>' : ""}
      <small>${desc} ${disabled ? "Indisponible sans carte graphique." : `≈ ${size >= 1000 ? (size / 1000).toFixed(1).replace(".", ",") + " Go" : size + " Mo"} à télécharger la première fois.`}</small></span>
    </label>`;
  });
  $("models").innerHTML = items.join("");
  $("models").onchange = updateResumeHint;
}

const selectedModel = () => document.querySelector('input[name="model"]:checked')?.value || "base";

// ---------------------------------------------------------------- choix du fichier

async function audioElementDuration(file) {
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve) => {
      const a = new Audio();
      a.preload = "metadata";
      a.onloadedmetadata = () => resolve(isFinite(a.duration) ? a.duration : null);
      a.onerror = () => resolve(null);
      a.src = url;
      setTimeout(() => resolve(null), 8000);
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

let chooseSeq = 0;
async function chooseFile(file) {
  if (!file || state.running) return;
  const seq = ++chooseSeq;
  const stale = () => seq !== chooseSeq || state.running;
  state.file = file;
  state.info = null;
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = "Analyse du fichier…";
  $("drop").classList.add("has-file");
  $("start").disabled = true;
  let info = { path: "whole", duration: null };
  if (state.caps.webcodecs || /\.wav$/i.test(file.name)) {
    const probe = new WorkerClient("./decoder-worker.js");
    try {
      const r = await probe.call("probe", { file });
      if (r.ok) info = { path: "stream", duration: r.duration, codec: r.codec };
      else if (r.reason === "NO_AUDIO") info = { path: "none" };
    } catch (e) {
      if (DEBUG) console.warn("probe", e);
    } finally {
      probe.terminate();
    }
  }
  if (stale()) return;
  if (info.path === "none") {
    $("fileMeta").textContent = "Ce fichier ne contient pas de piste audio.";
    return;
  }
  if (info.path === "whole") info.duration = await audioElementDuration(file);
  if (stale()) return;
  state.info = info;
  const mb = (file.size / 1048576).toFixed(file.size > 104857600 ? 0 : 1).replace(".", ",");
  let meta = `${mb} Mo` + (info.duration ? ` · ${formatDuration(info.duration)}` : "");
  if (info.path === "whole" && file.size > 400 * 1048576) {
    meta += " · ⚠️ fichier très lourd pour ce navigateur : utilisez Google Chrome si possible.";
  }
  $("fileMeta").textContent = meta;
  $("start").disabled = false;
  await updateResumeHint();
}

async function updateResumeHint() {
  const box = $("resume");
  box.hidden = true;
  if (!state.file || state.running) return;
  const rec = await store.get(`${fileKey(state.file)}|${selectedModel()}`);
  if (rec && !rec.done && rec.seek > 0) {
    const pct = rec.duration ? Math.round((100 * rec.seek) / rec.duration) : null;
    $("resumeText").textContent = `Une transcription de ce fichier s'est arrêtée${pct != null ? ` à ${pct} %` : ""}. Reprendre là où elle s'était arrêtée`;
    $("resumeCheck").checked = true;
    box.hidden = false;
  }
}

// ---------------------------------------------------------------- transcription

let wakeLock = null;
async function keepAwake() {
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
}
document.addEventListener("visibilitychange", () => {
  if (!state.running) return;
  if (document.visibilityState === "visible") {
    keepAwake();
  } else {
    $("hiddenWarning").hidden = false; // affiché au retour sur l'onglet
  }
});
window.addEventListener("beforeunload", (e) => {
  if (state.running) { e.preventDefault(); e.returnValue = ""; }
});

function setStage(text) { $("stage").textContent = text; }

function setProgress(fraction, detail) {
  const pct = Math.max(0, Math.min(100, fraction * 100));
  $("bar").style.width = pct.toFixed(1) + "%";
  $("pct").textContent = Math.floor(pct) + " %";
  if (detail != null) $("detail").textContent = detail;
}

function appendLive(segments) {
  const live = $("live");
  const atBottom = live.scrollHeight - live.scrollTop - live.clientHeight < 40;
  for (const s of segments) {
    const p = document.createElement("p");
    p.innerHTML = `<time>${clock(s.start)}</time> `;
    p.append(document.createTextNode(s.text));
    live.append(p);
  }
  while (live.childElementCount > 400) live.firstElementChild.remove();
  if (atBottom) live.scrollTop = live.scrollHeight;
}

const clock = (t) => {
  const s = Math.floor(t), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return (h ? `${h}:` : "") + `${String(m).padStart(h ? 2 : 1, "0")}:${String(s % 60).padStart(2, "0")}`;
};

async function createAsr(spec, onProgress) {
  const client = new WorkerClient("./asr-worker.js", (ev) => {
    if (ev.type === "progress") onProgress?.(ev);
  });
  state.asr = client;
  if (state.cancel) client.terminate();
  try {
    await client.call("load", {
      model: spec.id, device: spec.device, dtype: spec.dtype, debug: DEBUG, leakFix: !params.has("noleakfix"),
    });
    return client;
  } catch (e) {
    client.terminate();
    throw e;
  }
}

function downloadProgress(ev) {
  if (ev.status === "progress_total" && ev.total) {
    const mbDone = Math.round(ev.loaded / 1048576), mbTotal = Math.round(ev.total / 1048576);
    setProgress(ev.loaded / ev.total, `${mbDone} / ${mbTotal} Mo (téléchargé une seule fois, ensuite gardé sur ce Mac)`);
  }
}

const NETWORK_ERROR = /fetch|network|Failed to load|NetworkError|load failed|Could not locate|ERR_|s'est arrêté/i;

/** Charge le moteur. `quiet` : rechargement en cours de route (le modèle vient du cache). */
async function loadAsr(spec, { quiet = false } = {}) {
  if (!quiet) {
    setStage("Chargement du modèle de reconnaissance vocale…");
    setProgress(0, "");
  }
  const onProgress = quiet ? null : downloadProgress;
  try {
    return { client: await createAsr(spec, onProgress), spec };
  } catch (e) {
    // Problème de réseau : inutile de changer de moteur, on laisse l'erreur remonter.
    if (spec.device !== "webgpu" || state.cancel || NETWORK_ERROR.test(e.message)) throw e;
    // La carte graphique a échoué : on retente sur le processeur, pour cette transcription seulement.
    console.warn("WebGPU indisponible, repli WebAssembly", e);
    notice(`La carte graphique n'a pas pu être utilisée (${e.message}). Passage au processeur, plus lent.`);
    const fallback = modelSpec(spec.key, { forceCpu: true });
    return { client: await createAsr(fallback, onProgress), spec: fallback };
  }
}

function notice(text) {
  const n = $("notice");
  n.textContent = text;
  n.hidden = !text;
}

async function run() {
  const file = state.file;
  const info = state.info;
  const modelKey = selectedModel();
  const key = `${fileKey(file)}|${modelKey}`;
  const languageChoice = $("language").value;
  const multilingual = $("multilingual").checked;
  // Reprise automatique d'un travail interrompu, sauf si l'utilisateur a décoché la case.
  const rec = await store.get(key);
  const declined = !$("resume").hidden && !$("resumeCheck").checked;
  let saved = rec && !rec.done && rec.seek > 0 && !declined ? rec : null;
  let restartNotice = "";
  if (saved && (!!saved.multilingual !== multilingual ||
      (languageChoice !== "auto" && !multilingual && saved.language !== languageChoice))) {
    saved = null; // réglages de langue changés : on repart du début
    restartNotice = "Réglage de langue modifié : la transcription repart du début.";
  }
  if (rec?.done) {
    // Une transcription terminée de ce fichier est conservée à part dans l'historique.
    await store.put({ ...rec, key: `${key}|${rec.updatedAt}` });
    await store.del(key);
  }

  state.running = true;
  state.cancel = false;
  state.result = null;
  document.body.classList.add("running");
  $("result").hidden = true;
  $("progressCard").hidden = false;
  $("hiddenWarning").hidden = true;
  $("live").innerHTML = "";
  notice(restartNotice);
  keepAwake();
  $("progressCard").scrollIntoView({ behavior: "smooth", block: "start" });

  let source = null, asr = null, spec = modelSpec(modelKey);
  const segments = saved?.segments ? [...saved.segments] : [];
  let seekS = saved?.seek || 0;
  let language = saved?.language || (languageChoice === "auto" ? null : languageChoice);
  let languageProb = saved?.languageProb ?? null;
  let duration = info.duration || saved?.duration || null;
  if (segments.length) appendLive(segments.slice(-50));

  const record = () => ({
    key, name: file.name, size: file.size, lastModified: file.lastModified,
    modelKey, model: spec.id, duration, seek: seekS, language, languageProb, multilingual,
    segments, done: false, updatedAt: Date.now(),
  });

  try {
    // 1. Ouverture du fichier
    setStage("Lecture du fichier audio…");
    if (info.path === "stream") {
      source = new StreamSource(file);
    } else {
      setProgress(0, "Décodage complet du fichier (peut prendre une minute)…");
      source = await WholeSource.create(file);
      duration = source.duration;
    }
    await source.open(seekS);
    const checkCancel = () => { if (state.cancel) throw new Error("CANCELLED"); };
    checkCancel();
    const buffer = new PcmBuffer(Math.round(seekS * SR));
    let eof = false;
    const fill = async (untilSample) => {
      while (!eof && buffer.end < untilSample) {
        const { pcm, eof: done, damaged } = await source.pull(60);
        checkCancel();
        buffer.append(pcm);
        eof = done;
        if (damaged && !damagedWarned) {
          damagedWarned = true;
          notice("Une partie du fichier est abîmée : elle a été sautée.");
        }
      }
    };
    let damagedWarned = false;

    // 2. Modèle
    ({ client: asr, spec } = await loadAsr(spec));
    if (state.cancel) throw new Error("CANCELLED");

    // 3. Langue
    if (!language && !multilingual) {
      setStage("Détection de la langue…");
      setProgress(0, "");
      // Les 5 passages les plus parlés des 10 premières minutes votent.
      await fill(buffer.start + 600 * SR);
      const windows = [];
      for (let s = buffer.start; s < buffer.end; s += WINDOW_S * SR) {
        const pcm = buffer.slice(s, s + WINDOW_S * SR);
        windows.push({ pcm, full: pcm.length === WINDOW_S * SR });
      }
      const chosen = pickLanguageWindows(windows, 5);
      const votes = [];
      for (const [i, w] of chosen.entries()) {
        setProgress(i / chosen.length, `Écoute du passage ${i + 1} sur ${chosen.length}…`);
        const r = await asr.call("detect", { pcm: w.pcm }, [], windowTimeout(0));
        votes.push(r.probs);
      }
      const ranking = combineLanguageVotes(votes);
      [language, languageProb] = ranking[0] || ["en", null];
      if (DEBUG) console.log("langues", JSON.stringify(votes.map((v) => v.slice(0, 3))), JSON.stringify(ranking.slice(0, 5)));
    }
    $("langInfo").textContent = multilingual
      ? "Langue : détectée à chaque passage (mode multilingue)"
      : `Langue : ${langName(language)}${languageProb != null ? ` (sûr à ${Math.round(languageProb * 100)} %)` : ""}`;
    state.language = language;
    $("langFix").hidden = multilingual;
    if (!multilingual && languageProb != null && languageProb < 0.7) {
      notice(`La langue n'est pas certaine. Si le texte qui apparaît n'est pas en ${langName(language).toLowerCase()}, cliquez sur « Pas la bonne langue ? ».`);
    }

    // 4. Transcription fenêtre par fenêtre
    setStage("Transcription en cours…");
    const t0 = performance.now();
    const seek0 = seekS;
    let windowsDone = 0;
    let slowWarned = false;
    let avgMs = 0; // durée moyenne d'une fenêtre (pour le délai de garde)
    let saveWarned = false;
    while (!state.cancel) {
      const startSample = Math.round(seekS * SR);
      await fill(startSample + WINDOW_S * SR);
      const endSample = Math.min(startSample + WINDOW_S * SR, buffer.end);
      if (endSample - startSample < SR / 5) break; // moins de 0,2 s restant
      const isLast = eof && endSample === buffer.end;
      const pcm = buffer.slice(startSample, endSample);
      const lenS = pcm.length / SR;

      let advance = lenS;
      if (!isSilent(pcm)) {
        let lang = language;
        if (multilingual) {
          const r = await asr.call("detect", { pcm }, [], windowTimeout(avgMs));
          lang = r.probs[0][0];
        }
        let out;
        try {
          out = await asr.call("transcribe", { pcm, language: lang, simple: params.has("simple") }, [], windowTimeout(avgMs));
        } catch (e) {
          if (state.cancel) throw e;
          // Le moteur a planté ou bloqué (mémoire GPU, boucle de la bibliothèque…) : on le
          // redémarre et on retente une fois en mode simple (un seul passage, longueur bornée).
          console.warn("Nouvel essai après erreur", e);
          asr.terminate();
          setStage("Redémarrage du moteur…");
          ({ client: asr, spec } = await loadAsr(spec, { quiet: true }));
          setStage("Transcription en cours…");
          out = await asr.call("transcribe", { pcm, language: lang, simple: true }, [], windowTimeout(avgMs));
        }
        avgMs = avgMs ? 0.8 * avgMs + 0.2 * out.ms : out.ms;
        const merged = mergeWindow(out.chunks, seekS, lenS, isLast);
        const fresh = [];
        for (const s of merged.segments) {
          if (isHallucination(s.text) || isRepeatLoop(s.text, segments)) continue;
          const seg = multilingual ? { ...s, language: lang } : s;
          segments.push(seg);
          fresh.push(seg);
        }
        appendLive(fresh);
        advance = merged.advance;
        if (DEBUG) {
          $("debug").textContent = `fenêtre ${windowsDone + 1} · ${Math.round(out.ms)} ms · GPU ${(out.gpu / 1048576).toFixed(0)} Mo (pic ${(out.gpuPeak / 1048576).toFixed(0)} Mo)`;
          console.log("fenêtre", windowsDone + 1, out);
        }
      }
      seekS = (startSample + Math.round(advance * SR)) / SR;
      buffer.dropBefore(Math.round(seekS * SR));
      windowsDone++;

      // Progression, vitesse, temps restant
      const elapsed = (performance.now() - t0) / 1000;
      const speed = (seekS - seek0) / Math.max(elapsed, 0.001);
      const total = duration || Math.max(seekS, 1);
      const remaining = speed > 0 ? (total - seekS) / speed : null;
      setProgress(seekS / total,
        `${formatDuration(seekS)} sur ${formatDuration(total)} · ` +
        (elapsed > 20 && remaining != null ? `encore ≈ ${formatDuration(remaining)} · vitesse ×${speed.toFixed(1).replace(".", ",")}` : "estimation du temps restant…"));
      if (!slowWarned && elapsed > 120 && remaining > 4 * 3600) {
        slowWarned = true;
        notice("Ce Mac est lent pour ce modèle : essayez le mode « Équilibré » ou « Rapide », ou la version Google Colab (plus bas).");
      }
      await store.put(record());

      if (isLast && seekS >= endSample / SR - 0.01) break;
      if (windowsDone % RECYCLE_WINDOWS === 0) {
        // Libère complètement la mémoire GPU en recréant le moteur (le modèle vient du cache).
        asr.terminate();
        setStage("Rafraîchissement du moteur (quelques secondes)…");
        ({ client: asr, spec } = await loadAsr(spec, { quiet: true }));
        setStage("Transcription en cours…");
      }
      if (store.failed && !saveWarned) {
        saveWarned = true;
        notice("La sauvegarde automatique ne fonctionne pas (disque plein ou navigation privée) : gardez cette page ouverte jusqu'à la fin.");
      }
    }
    if (state.cancel) throw new Error("CANCELLED");

    duration = duration || seekS;
    const result = {
      file: file.name,
      model: spec.id,
      language: multilingual ? mostFrequentLanguage(segments) : language,
      language_probability: multilingual ? null : languageProb,
      duration,
      segments,
    };
    await store.put({ ...record(), seek: duration, done: true, result });
    showResult(result);
    setStage("Terminé !");
    setProgress(1, `Transcription terminée en ${formatDuration((performance.now() - t0) / 1000)}.`);
    renderHistory();
  } catch (e) {
    if (state.restart) {
      // Relance demandée (changement de langue) : pas de message d'arrêt.
    } else if (e.message === "CANCELLED" || state.cancel) {
      setStage("Transcription arrêtée.");
      notice("Le travail déjà fait est gardé : cliquez sur « Lancer la transcription » pour reprendre.");
    } else {
      console.error(e);
      setStage("Une erreur est survenue.");
      notice(friendlyError(e));
    }
  } finally {
    asr?.terminate();
    source?.close();
    state.running = false;
    document.body.classList.remove("running");
    try { await wakeLock?.release(); } catch {}
    wakeLock = null;
    $("langFix").hidden = true;
    $("langPick").hidden = true;
    if (state.restart) {
      // Recommence depuis le début avec la langue choisie par l'utilisateur.
      const { language: forced } = state.restart;
      state.restart = null;
      await store.del(key);
      $("language").value = forced;
      $("multilingual").checked = false;
      $("resume").hidden = true;
      run();
    } else {
      // Montre la case « Reprendre » et l'historique à jour pour le prochain clic.
      renderHistory();
      await updateResumeHint();
    }
  }
}

function mostFrequentLanguage(segments) {
  const n = new Map();
  for (const s of segments) if (s.language) n.set(s.language, (n.get(s.language) || 0) + (s.end - s.start));
  return [...n.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null;
}

function friendlyError(e) {
  const m = String(e?.message || e);
  if (/fetch|network|Failed to load|NetworkError|load failed/i.test(m))
    return "Le téléchargement a échoué. Vérifiez la connexion Internet puis relancez : ce qui a déjà été transcrit est gardé.";
  if (/UNREADABLE|decode|EncodingError|CANNOT_DECODE/i.test(m))
    return "Ce fichier n'a pas pu être lu. Essayez un autre format (MP3, M4A, WAV) ou Google Chrome.";
  if (/memory|allocation|OOM|device lost|lost/i.test(m))
    return "Le Mac a manqué de mémoire. Fermez les autres applications et relancez (la transcription reprendra où elle s'était arrêtée), ou choisissez un mode plus léger.";
  if (/TIMEOUT/.test(m) || /CANCELLED/.test(m))
    return "Le moteur ne répond plus. Relancez : la transcription reprendra où elle s'était arrêtée.";
  return `Erreur : ${m}. Relancez : la transcription reprendra où elle s'était arrêtée.`;
}

// ---------------------------------------------------------------- résultats

function baseName(name) { return name.replace(/\.[^.]+$/, "") || "transcription"; }

function download(result, fmt) {
  const blob = new Blob([FORMATS[fmt](result)], { type: `${MIME[fmt]};charset=utf-8` });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${baseName(result.file)}.${fmt}`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

function showResult(result) {
  state.result = result;
  $("result").hidden = false;
  $("resultMeta").textContent =
    `${result.file} · ${formatDuration(result.duration)} · langue : ${langName(result.language)}`;
  $("resultText").textContent = FORMATS.txt(result);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function renderHistory() {
  const all = ((await store.all()) || []).sort((a, b) => b.updatedAt - a.updatedAt);
  const list = $("history");
  $("historyCard").hidden = all.length === 0;
  list.innerHTML = "";
  for (const rec of all.slice(0, 30)) {
    const li = document.createElement("li");
    const pct = rec.duration ? Math.min(100, Math.round((100 * rec.seek) / rec.duration)) : 0;
    const when = new Date(rec.updatedAt).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
    li.innerHTML = `<div><strong></strong><small>${when} · ${MODELS[rec.modelKey]?.name || rec.modelKey} · ${rec.done ? "terminé" : `interrompu à ${pct} %`}</small></div><div class="actions"></div>`;
    li.querySelector("strong").textContent = rec.name;
    const actions = li.querySelector(".actions");
    const result = rec.result || {
      file: rec.name, model: rec.model, language: rec.language, duration: rec.duration, segments: rec.segments,
    };
    for (const fmt of ["txt", "srt"]) {
      const b = document.createElement("button");
      b.className = "small";
      b.textContent = fmt.toUpperCase();
      b.title = rec.done ? "" : "Télécharger la partie déjà transcrite";
      b.onclick = () => download(result, fmt);
      actions.append(b);
    }
    const del = document.createElement("button");
    del.className = "small ghost";
    del.textContent = "Supprimer";
    del.onclick = async () => { await store.del(rec.key); renderHistory(); updateResumeHint(); };
    actions.append(del);
    list.append(li);
  }
}

// ---------------------------------------------------------------- initialisation

async function init() {
  $("colab").href = COLAB_URL;
  $("colab2").href = COLAB_URL;
  if (DEBUG) $("debug").hidden = false;
  state.caps = await detectCapabilities();
  if (params.get("device") === "wasm") state.caps.webgpu = false;
  renderCapabilities(state.caps);
  renderModelChoices();
  renderHistory();
  navigator.storage?.persist?.()?.catch?.(() => {}); // sans attendre : ne doit pas bloquer la page

  const drop = $("drop"), input = $("file");
  drop.onclick = () => !state.running && input.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
  input.onchange = () => {
    const f = input.files[0];
    input.value = ""; // permet de rechoisir le même fichier
    chooseFile(f);
  };
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => chooseFile(e.dataTransfer.files[0]));

  $("start").onclick = () => !state.running && state.file && state.info && run();
  // Correction de la langue pendant la transcription.
  $("langSelect").innerHTML = [...$("language").options]
    .filter((o) => o.value !== "auto")
    .map((o) => `<option value="${o.value}">${o.textContent}</option>`)
    .join("");
  $("langFix").onclick = () => {
    if (state.language && [...$("langSelect").options].some((o) => o.value === state.language)) {
      $("langSelect").value = state.language;
    }
    $("langFix").hidden = true;
    $("langPick").hidden = false;
  };
  $("langApply").onclick = () => {
    if (!state.running) return;
    state.restart = { language: $("langSelect").value };
    state.cancel = true;
    setStage("Redémarrage dans la langue choisie…");
    state.asr?.terminate();
  };
  $("cancel").onclick = () => {
    if (!state.running) return;
    state.cancel = true;
    setStage("Arrêt en cours…");
    state.asr?.terminate(); // interrompt aussi un téléchargement ou une fenêtre en cours
  };
  for (const fmt of Object.keys(FORMATS)) {
    const b = $(`dl-${fmt}`);
    if (b) b.onclick = () => state.result && download(state.result, fmt);
  }
  $("copy").onclick = async () => {
    if (!state.result) return;
    try {
      await navigator.clipboard.writeText(FORMATS.txt(state.result));
      $("copy").textContent = "Copié ✓";
      setTimeout(() => ($("copy").textContent = "Copier le texte"), 2000);
    } catch {}
  };
  document.body.classList.add("ready");
}

init();
