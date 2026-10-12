// Export des transcriptions : TXT, SRT, VTT, JSON (même logique que transcriber/formats.py).

function ts(seconds, sep) {
  let ms = Math.max(0, Math.round(seconds * 1000));
  const h = Math.floor(ms / 3600000); ms -= h * 3600000;
  const m = Math.floor(ms / 60000); ms -= m * 60000;
  const s = Math.floor(ms / 1000); ms -= s * 1000;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)}${sep}${p(ms, 3)}`;
}

const nonEmpty = (segments) => segments.filter((s) => s.text && s.text.trim());

export function toTxt(t) {
  return nonEmpty(t.segments).map((s) => s.text.trim()).join("\n") + "\n";
}

export function toSrt(t) {
  return nonEmpty(t.segments)
    .map((s, i) => `${i + 1}\n${ts(s.start, ",")} --> ${ts(s.end, ",")}\n${s.text.trim()}\n`)
    .join("\n");
}

export function toVtt(t) {
  const lines = ["WEBVTT", ""];
  for (const s of nonEmpty(t.segments)) lines.push(`${ts(s.start, ".")} --> ${ts(s.end, ".")}`, s.text.trim(), "");
  return lines.join("\n");
}

export function toJson(t) {
  return JSON.stringify({ ...t, text: toTxt(t).trimEnd() }, null, 2);
}

export const FORMATS = { txt: toTxt, srt: toSrt, vtt: toVtt, json: toJson };
export const MIME = { txt: "text/plain", srt: "application/x-subrip", vtt: "text/vtt", json: "application/json" };
