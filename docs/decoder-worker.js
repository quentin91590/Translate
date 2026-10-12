// Worker de décodage : lit le fichier audio/vidéo EN FLUX (WebCodecs via Mediabunny)
// et fournit du PCM 16 kHz mono à la demande. La mémoire reste constante, même pour 3 h :
// on ne décode jamais tout le fichier d'un coup.
import { Input, BlobSource, ALL_FORMATS, AudioSampleSink } from "./vendor/mediabunny.min.mjs";
import { StreamingResampler } from "./resampler.js";

const SR = 16000;
let input = null;
let iterator = null;
let resampler = null;
let carry = new Float32Array(0); // échantillons déjà rééchantillonnés, pas encore envoyés
let trimFrames = 0; // échantillons source à ignorer au début (reprise en milieu de fichier)
let eof = false;
let startOffset = 0; // instant (s) où commence la lecture
let mono = new Float32Array(0);
let plane = new Float32Array(0);

function close() {
  try { iterator?.return?.(); } catch {}
  try { input?.dispose(); } catch {}
  input = iterator = resampler = null;
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

async function probe(file) {
  const { inp, track, error } = await openTrack(file);
  try {
    if (error) return { ok: false, reason: error };
    const codec = await track.getCodec();
    const canDecode = await track.canDecode();
    let duration = await inp.getDurationFromMetadata([track]);
    if (!duration) duration = await inp.computeDuration([track]);
    return {
      ok: canDecode,
      reason: canDecode ? null : "CANNOT_DECODE",
      codec,
      duration,
      sampleRate: track.sampleRate,
      channels: track.numberOfChannels,
    };
  } finally {
    inp.dispose();
  }
}

async function open(file, startSec) {
  close();
  const { inp, track, error } = await openTrack(file);
  if (error) { inp.dispose(); throw new Error(error); }
  input = inp;
  iterator = new AudioSampleSink(track).samples(startSec > 0 ? startSec : undefined)[Symbol.asyncIterator]();
  trimFrames = startSec > 0 ? null : 0; // calculé sur le premier échantillon reçu
  startOffset = startSec;
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

async function pull(seconds) {
  const want = Math.round(seconds * SR);
  const parts = [carry];
  let have = carry.length;
  while (have < want && !eof) {
    const { value: sample, done } = await iterator.next();
    if (done) {
      eof = true;
      if (resampler) { const tail = resampler.flush(); parts.push(tail); have += tail.length; }
      break;
    }
    try {
      resampler ??= new StreamingResampler(sample.sampleRate, SR);
      let m = toMono(sample);
      if (trimFrames === null) {
        // Reprise : le premier bloc commence un peu avant l'instant demandé.
        trimFrames = Math.max(0, Math.round((startOffset - sample.timestamp) * sample.sampleRate));
      }
      if (trimFrames > 0) {
        const skip = Math.min(trimFrames, m.length);
        m = m.subarray(skip);
        trimFrames -= skip;
      }
      if (m.length) {
        const out = resampler.push(m);
        parts.push(out);
        have += out.length;
      }
    } finally {
      sample.close(); // libère l'AudioData décodé
    }
  }
  const all = concatAll(parts);
  const pcm = all.length > want ? all.slice(0, want) : all;
  carry = all.length > want ? all.slice(want) : new Float32Array(0);
  return { pcm, eof: eof && carry.length === 0 };
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
