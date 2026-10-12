// Logique pure (sans navigateur) : fenêtres de 30 s, fusion des segments, filtres.
// Testée dans tests/web/core.test.mjs.

export const SR = 16000;          // Whisper attend du 16 kHz mono
export const WINDOW_S = 30;       // taille de la fenêtre d'entrée de Whisper

/**
 * Convertit la sortie du pipeline pour une fenêtre en segments absolus et calcule
 * de combien avancer (méthode « seek » d'OpenAI) :
 *  - les segments complets (horodatage de fin présent) sont gardés ;
 *  - un dernier segment inachevé (fin = null) est jeté et la fenêtre suivante
 *    repart de la fin du dernier segment complet, pour ne pas couper une phrase ;
 *  - sans aucun segment complet, toute la fenêtre est prise telle quelle.
 *
 * @param {{text: string, timestamp: [number|null, number|null]}[]} chunks
 * @param {number} startS début de la fenêtre dans le fichier (s)
 * @param {number} lenS durée de la fenêtre (s)
 * @param {boolean} isLast vrai si la fenêtre va jusqu'à la fin du fichier
 * @returns {{segments: {start: number, end: number, text: string}[], advance: number}}
 */
export function mergeWindow(chunks, startS, lenS, isLast) {
  const clamp = (t) => Math.min(Math.max(t, 0), lenS);
  const complete = [];
  let pending = null;
  for (const c of chunks || []) {
    const [s, e] = c.timestamp || [null, null];
    if (e == null) {
      pending = { s: s ?? (complete.at(-1)?.e ?? 0), text: c.text };
      break; // seul le dernier morceau peut être inachevé
    }
    complete.push({ s: clamp(s ?? 0), e: clamp(e), text: c.text });
  }

  const lastEnd = complete.length ? complete.at(-1).e : 0;
  let advance = lenS;
  const kept = complete;
  if (pending) {
    if (isLast || lastEnd < 1) {
      // Fin de fichier, ou rien de complet : on garde le texte jusqu'à la fin de la fenêtre.
      kept.push({ s: clamp(pending.s), e: lenS, text: pending.text });
    } else {
      advance = lastEnd; // on reprendra au début de la phrase inachevée
    }
  }
  const segments = [];
  for (const k of kept) {
    const text = (k.text || "").trim();
    if (!text) continue;
    const s = Math.min(k.s, k.e);
    segments.push({ start: round3(startS + s), end: round3(startS + k.e), text });
  }
  return { segments, advance };
}

const round3 = (x) => Math.round(x * 1000) / 1000;

// Phrases que Whisper « invente » sur du silence ou de la musique (génériques de sous-titres).
const HALLUCINATIONS = [
  /amara\.org/i,
  /^sous-titr(age|es|é)/i,
  /^(merci d'avoir regardé|thank(s| you) for watching|please subscribe|abonnez-vous)/i,
  /^untertitel (der|im auftrag|von)/i,
  /^subt[ií]tulos (realizados|por)/i,
  /^sottotitoli (creati|a cura)/i,
  /^ondertitel(d|s)? door/i,
  /^legendas pela/i,
];

/** Vrai si le segment ressemble à une hallucination connue ou à une boucle de répétition. */
export function isHallucination(text) {
  const t = (text || "").trim();
  if (!/[\p{L}\p{N}]/u.test(t)) return true; // vide ou seulement de la ponctuation (« ... »)
  if (t.length < 120 && HALLUCINATIONS.some((re) => re.test(t))) return true;
  const words = t.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length >= 16) {
    const unique = new Set(words).size;
    if (unique / words.length < 0.2) return true; // « la la la la… »
  }
  // Même caractère ou syllabe répété en continu (ex. « hmhmhmhmhm… » ou « ーーーー »).
  if (/(.{1,4})\1{12,}/u.test(t.replace(/\s+/g, ""))) return true;
  return false;
}

/** Niveau sonore maximal par trames de 0,5 s (RMS linéaire). */
export function frameLevels(pcm, frame = SR / 2) {
  const out = [];
  for (let i = 0; i < pcm.length; i += frame) {
    const end = Math.min(i + frame, pcm.length);
    let sum = 0;
    for (let j = i; j < end; j++) sum += pcm[j] * pcm[j];
    out.push(Math.sqrt(sum / Math.max(1, end - i)));
  }
  return out;
}

const DB = (db) => Math.pow(10, db / 20);

/** Vrai si la fenêtre ne contient aucun son au-dessus de -50 dBFS : on peut la sauter. */
export function isSilent(pcm) {
  const levels = frameLevels(pcm);
  return levels.length === 0 || Math.max(...levels) < DB(-50);
}

/** Part des trames de 0,5 s avec un niveau « parole probable » (> -40 dBFS). */
export function speechScore(pcm) {
  const levels = frameLevels(pcm);
  if (!levels.length) return 0;
  return levels.filter((l) => l > DB(-40)).length / levels.length;
}

/** Additionne les probabilités de langue de plusieurs fenêtres et renvoie le classement. */
export function combineLanguageVotes(votes) {
  const total = new Map();
  for (const v of votes) for (const [code, p] of v) total.set(code, (total.get(code) || 0) + p);
  const sum = [...total.values()].reduce((a, b) => a + b, 0) || 1;
  return [...total.entries()].map(([c, p]) => [c, p / sum]).sort((a, b) => b[1] - a[1]);
}

/** Durée lisible en français : « 2 h 05 », « 12 min », « 45 s ». */
export function formatDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h} h ${String(m).padStart(2, "0")}`;
  if (m) return `${m} min${s % 60 && m < 10 ? ` ${String(s % 60).padStart(2, "0")} s` : ""}`;
  return `${s} s`;
}

/** Tampon PCM glissant : on ajoute à la fin, on lit des tranches, on oublie le début. */
export class PcmBuffer {
  constructor(startSample = 0) {
    this.start = startSample; // index (dans le fichier) du premier échantillon conservé
    this.data = new Float32Array(SR * 120);
    this.length = 0;
  }
  get end() {
    return this.start + this.length;
  }
  append(pcm) {
    if (this.length + pcm.length > this.data.length) {
      const bigger = new Float32Array(Math.max(this.data.length * 2, this.length + pcm.length));
      bigger.set(this.data.subarray(0, this.length));
      this.data = bigger;
    }
    this.data.set(pcm, this.length);
    this.length += pcm.length;
  }
  /** Copie des échantillons [from, to[ (indices absolus). */
  slice(from, to) {
    const a = Math.max(0, from - this.start), b = Math.min(this.length, to - this.start);
    return this.data.slice(a, Math.max(a, b));
  }
  dropBefore(sample) {
    const n = Math.min(this.length, Math.max(0, sample - this.start));
    if (n <= 0) return;
    this.data.copyWithin(0, n, this.length);
    this.length -= n;
    this.start += n;
  }
}
