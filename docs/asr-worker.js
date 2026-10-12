// Worker de reconnaissance vocale : Whisper via Transformers.js (WebGPU, sinon WebAssembly).
// Reçoit des fenêtres de 30 s en 16 kHz mono et renvoie le texte horodaté.
// Version figée : le correctif de fuite mémoire ci-dessous dépend de son code interne.
import { pipeline, env, LogitsProcessor, softmax } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";

env.allowLocalModels = false; // modèles téléchargés depuis huggingface.co puis gardés en cache
env.useBrowserCache = true;

let asr = null;
const gpuStats = { live: 0, peak: 0 };

/** Comptabilise la mémoire GPU allouée (mode diagnostic ?debug=1). */
function trackGpuMemory() {
  if (typeof GPUDevice === "undefined" || GPUDevice.prototype.__tracked) return;
  const sizes = new WeakMap();
  const create = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function (desc) {
    const buf = create.call(this, desc);
    sizes.set(buf, desc.size);
    gpuStats.live += desc.size;
    gpuStats.peak = Math.max(gpuStats.peak, gpuStats.live);
    return buf;
  };
  const destroy = GPUBuffer.prototype.destroy;
  GPUBuffer.prototype.destroy = function () {
    const size = sizes.get(this);
    if (size !== undefined) { gpuStats.live -= size; sizes.delete(this); }
    return destroy.call(this);
  };
  GPUDevice.prototype.__tracked = true;
}

/**
 * Corrige la fuite de mémoire GPU de Whisper sur WebGPU (transformers.js#1739, correctif
 * amont #1755 pas encore publié) : à chaque pas de décodage, le décodeur renvoie une nouvelle
 * copie GPU du cache de l'encodeur (`present.*.encoder.*`) que la bibliothèque n'utilise pas
 * et ne libère jamais — environ 0,5 Go perdu par fenêtre de 30 s. On libère ces doublons.
 */
function fixWebGpuLeak(model) {
  const forward = model.forward.bind(model);
  model.forward = async (inputs) => {
    const out = await forward(inputs);
    const past = inputs?.past_key_values;
    if (past) {
      for (const name in out) {
        if (!name.startsWith("present") || !name.includes("encoder")) continue;
        const fresh = out[name];
        const kept = past[name.replace("present", "past_key_values")];
        // Le cache conserve « kept » ; « fresh » est un doublon jamais relu.
        if (kept && fresh && fresh !== kept && fresh.location === "gpu-buffer") fresh.dispose();
      }
    }
    return out;
  };
}

/** Récupère les scores des jetons de langue au premier pas de décodage. */
class LanguageProbe extends LogitsProcessor {
  constructor(tokenIds) {
    super();
    this.tokenIds = tokenIds;
    this.scores = null;
  }
  _call(inputIds, logits) {
    const row = logits[0].data;
    this.scores = softmax(this.tokenIds.map((id) => row[id]));
    return logits;
  }
}

async function load({ model, device, dtype, debug, leakFix = true }) {
  if (debug) trackGpuMemory();
  asr = await pipeline("automatic-speech-recognition", model, {
    device,
    dtype,
    progress_callback: (p) => {
      if (p.status === "progress_total" || p.status === "initiate" || p.status === "done") {
        self.postMessage({ type: "progress", status: p.status, file: p.file, progress: p.progress, loaded: p.loaded, total: p.total });
      }
    },
  });
  if (device === "webgpu" && leakFix) fixWebGpuLeak(asr.model);
  // Petit passage à vide : compile les shaders maintenant plutôt que sur la première fenêtre.
  await asr(new Float32Array(16000), { language: "en", task: "transcribe", return_timestamps: true });
  return { device, gpu: gpuStats.live };
}

/** Probabilités de chaque langue pour une fenêtre (≤ 30 s). */
async function detect({ pcm }) {
  const gc = asr.model.generation_config;
  const langs = Object.entries(gc.lang_to_id || {});
  if (!langs.length) return { probs: [["en", 1]] }; // modèle anglais uniquement
  const probe = new LanguageProbe(langs.map(([, id]) => id));
  const { input_features } = await asr.processor(pcm);
  await asr.model.generate({
    inputs: input_features,
    decoder_input_ids: [gc.decoder_start_token_id], // <|startoftranscript|> seul : le modèle prédit la langue
    max_new_tokens: 1,
    logits_processor: [probe],
  });
  const probs = langs
    .map(([token], i) => [token.replace(/^<\|/, "").replace(/\|>$/, ""), probe.scores[i]])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  return { probs, gpu: gpuStats.live };
}

async function transcribe({ pcm, language }) {
  const t0 = performance.now();
  const out = await asr(pcm, { language, task: "transcribe", return_timestamps: true });
  return {
    chunks: (out.chunks || []).map((c) => ({ text: c.text, timestamp: c.timestamp })),
    text: out.text,
    ms: performance.now() - t0,
    gpu: gpuStats.live,
    gpuPeak: gpuStats.peak,
  };
}

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    let result;
    if (type === "load") result = await load(data);
    else if (type === "detect") result = await detect(data);
    else if (type === "transcribe") result = await transcribe(data);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
