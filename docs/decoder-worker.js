// Worker de décodage : lit le fichier audio/vidéo EN FLUX (WebCodecs via Mediabunny)
// et fournit du PCM 16 kHz mono à la demande. La mémoire reste constante, même pour 3 h :
// on ne décode jamais tout le fichier d'un coup.
//
// Toutes les positions sont comptées depuis le début réel de la piste (certains fichiers
// commencent à un instant non nul). Un passage abîmé est remplacé par du silence pour que
// les horodatages restent justes, puis la lecture reprend un peu plus loin.
import { Input, BlobSource, ALL_FORMATS, AudioSampleSink } from "./vendor/mediabunny.min.mjs";
import { StreamingResampler } from "./resampler.js";

const SR = 16000;
const MAX_RECOVERIES = 8; // nombre maximal de reprises après un passage illisible
const MAX_GAP_S = 600; // un trou plus long n'est pas comblé (fichier très anormal)

let input = null;
let sink = null;
let iterator = null;
let resampler = null;
let resamplerRate = 0;
let carry = new Float32Array(0); // échantillons 16 kHz déjà prêts, pas encore envoyés
let cursor = 0; // instant (s, horloge du fichier) jusqu'où l'audio a été produit
let trackEnd = null; // fin de la piste (s, horloge du fichier)
let eof = false;
let recoveries = 0;
let damaged = false;
let trackStart = 0; // début de la piste sur l'horloge du fichier
let mono = new Float32Array(0);
let plane = new Float32Array(0);

function close() {
  try { iterator?.return?.(); } catch {}
  try { input?.dispose(); } catch {}
  input = sink = iterator = resampler = null;
  carry = new Float32Array(0);
  eof = false;
}

async function openTrack(file) {
  const inp = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  if (!(await inp.canRead())) return { inp, error: "UNREADABLE" };
  const track = await inp.getPrimaryAudioTrack();
  if (!track) return { inp, error: "NO_AUDIO" };
  return { inp, track };
}

/** Début et fin de la piste sur l'horloge du fichier. */
async function trackBounds(inp, track) {
  const start = (await track.getFirstTimestamp().catch(() => 0)) || 0;
  let end = await inp.getDurationFromMetadata([track]).catch(() => null);
  if (!end) end = await inp.computeDuration([track]).catch(() => null);
  return { start, end };
}

async function probe(file) {
  const { inp, track, error } = await openTrack(file);
  try {
    if (error) return { ok: false, reason: error };
    const codec = await track.getCodec();
    const canDecode = await track.canDecode();
    const { start, end } = await trackBounds(inp, track);
    return {
      ok: canDecode,
      reason: canDecode ? null : "CANNOT_DECODE",
      codec,
      duration: end != null ? Math.max(0, end - start) : null,
      sampleRate: track.sampleRate,
      channels: track.numberOfChannels,
    };
  } finally {
    inp.dispose();
  }
}

function startIterator(fromTime) {
  try { iterator?.return?.(); } catch {}
  iterator = sink.samples(fromTime)[Symbol.asyncIterator]();
}

async function open(file, startSec) {
  close();
  const { inp, track, error } = await openTrack(file);
  if (error) { inp.dispose(); throw new Error(error); }
  input = inp;
  const { start, end } = await trackBounds(inp, track);
  trackEnd = end;
  trackStart = start;
  sink = new AudioSampleSink(track);
  cursor = start + Math.max(0, startSec);
  recoveries = 0;
  damaged = false;
  startIterator(startSec > 0 ? cursor : undefined);
}

function toMono(sample) {
  const frames = sample.numberOfFrames, ch = sample.numberOfChannels;
  if (mono.length < frames) { mono = new Float32Array(frames); plane = new Float32Array(frames); }
  const m = mono.subarray(0, frames), p = plane.subarray(0, frames);
  m.fill(0);
  for (let c = 0; c < ch; c++) {
    sample.copyTo(p, { planeIndex: c, format: "f32-planar" });
    for (let i = 0; i < frames; i++) m[i] += p[i];
  }
  if (ch > 1) for (let i = 0; i < frames; i++) m[i] /= ch;
  return m;
}

function concatAll(parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Float32Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** Ajoute `seconds` de silence (après avoir vidé le rééchantillonneur). */
function pushSilence(parts, seconds) {
  if (resampler) { parts.push(resampler.flush()); resampler = null; }
  const n = Math.round(Math.min(seconds, MAX_GAP_S) * SR);
  if (n > 0) parts.push(new Float32Array(n));
  return n;
}

/** Après une erreur ou un arrêt prématuré : on saute un peu plus loin, de plus en plus. */
function recover() {
  if (recoveries >= MAX_RECOVERIES || (trackEnd != null && cursor >= trackEnd - 0.5)) return false;
  const step = [1, 1, 5, 5, 15, 30, 60, 120][recoveries++];
  damaged = true;
  startIterator(cursor + step);
  return true;
}

async function pull(seconds) {
  const want = Math.round(seconds * SR);
  const parts = [carry];
  let have = carry.length;
  let truncatedAt = null;
  damaged = false;
  while (have < want && !eof) {
    let next;
    try {
      next = await iterator.next();
    } catch (err) {
      console.warn("Passage illisible", cursor, err);
      if (recover()) continue;
      next = { done: true };
    }
    if (next.done) {
      // La piste s'arrête bien avant la fin annoncée : demuxeur bloqué sur une zone abîmée.
      if (trackEnd != null && trackEnd - cursor > 2) {
        if (recover()) continue;
        truncatedAt = cursor - trackStart; // impossible d'aller plus loin
      }
      eof = true;
      if (resampler) { const tail = resampler.flush(); parts.push(tail); have += tail.length; resampler = null; }
      break;
    }
    const sample = next.value;
    try {
      const rate = sample.sampleRate;
      const sStart = sample.timestamp;
      const sEnd = sStart + sample.numberOfFrames / rate;
      if (sEnd <= cursor + 1e-4) continue; // déjà produit (reprise qui recouvre)
      if (sStart > cursor + 0.05) {
        // Trou dans le fichier (ou passage sauté) : silence pour garder l'horloge juste.
        have += pushSilence(parts, sStart - cursor);
        cursor = sStart;
      }
      if (!resampler || resamplerRate !== rate) {
        if (resampler) { const tail = resampler.flush(); parts.push(tail); have += tail.length; }
        resampler = new StreamingResampler(rate, SR);
        resamplerRate = rate;
      }
      let m = toMono(sample);
      const skip = Math.max(0, Math.round((cursor - sStart) * rate)); // début déjà produit
      if (skip > 0) m = m.subarray(Math.min(skip, m.length));
      if (m.length) {
        const out = resampler.push(m);
        parts.push(out);
        have += out.length;
      }
      cursor = sEnd;
    } finally {
      sample.close(); // libère l'AudioData décodé
    }
  }
  const all = concatAll(parts);
  const pcm = all.length > want ? all.slice(0, want) : all;
  carry = all.length > want ? all.slice(want) : new Float32Array(0);
  return { pcm, eof: eof && carry.length === 0, damaged, truncatedAt: carry.length ? null : truncatedAt };
}

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    let result;
    if (type === "probe") result = await probe(data.file);
    else if (type === "open") result = await open(data.file, data.startSec || 0);
    else if (type === "pull") {
      result = await pull(data.seconds);
      self.postMessage({ id, ok: true, result }, [result.pcm.buffer]);
      return;
    } else if (type === "close") result = close();
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
