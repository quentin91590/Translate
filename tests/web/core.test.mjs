// Tests unitaires de la logique de la page web : node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SR, mergeWindow, isHallucination, isSilent, speechScore, combineLanguageVotes, formatDuration, PcmBuffer,
  pickLanguageWindows, isRepeatLoop,
} from "../../docs/core.js";
import { StreamingResampler } from "../../docs/resampler.js";
import { toSrt, toVtt, toTxt, toJson } from "../../docs/formats.js";

const ch = (text, s, e) => ({ text, timestamp: [s, e] });

test("mergeWindow: segments complets, avance de toute la fenêtre", () => {
  const r = mergeWindow([ch(" Bonjour.", 0, 4.2), ch(" Ça va ?", 4.2, 29.9)], 60, 30, false);
  assert.deepEqual(r.segments, [
    { start: 60, end: 64.2, text: "Bonjour." },
    { start: 64.2, end: 89.9, text: "Ça va ?" },
  ]);
  assert.equal(r.advance, 30);
});

test("mergeWindow: phrase inachevée reprise à la fenêtre suivante", () => {
  const r = mergeWindow([ch("Un.", 0, 10), ch("Deux.", 10, 21.5), ch("Trois qui continue", 21.5, null)], 0, 30, false);
  assert.equal(r.segments.length, 2);
  assert.equal(r.advance, 21.5);
});

test("mergeWindow: fin de fichier, le morceau inachevé est gardé", () => {
  const r = mergeWindow([ch("Un.", 0, 3), ch("Fin", 3, null)], 100, 12, true);
  assert.deepEqual(r.segments.at(-1), { start: 103, end: 112, text: "Fin" });
  assert.equal(r.advance, 12);
});

test("mergeWindow: aucun horodatage complet -> toute la fenêtre", () => {
  const r = mergeWindow([ch("Long monologue sans fin", 0, null)], 0, 30, false);
  assert.deepEqual(r.segments, [{ start: 0, end: 30, text: "Long monologue sans fin" }]);
  assert.equal(r.advance, 30);
});

test("mergeWindow: horodatages hors fenêtre ramenés dans la fenêtre", () => {
  const r = mergeWindow([ch("x", 2, 31.5)], 0, 25, true);
  assert.deepEqual(r.segments, [{ start: 2, end: 25, text: "x" }]);
});

test("mergeWindow: segments vides ignorés, sortie vide", () => {
  assert.deepEqual(mergeWindow([ch("  ", 0, 2)], 0, 30, false).segments, []);
  assert.deepEqual(mergeWindow([], 0, 30, false), { segments: [], advance: 30 });
});

test("isHallucination", () => {
  assert.ok(isHallucination("Sous-titres réalisés par la communauté d'Amara.org"));
  assert.ok(isHallucination("Merci d'avoir regardé cette vidéo !"));
  assert.ok(isHallucination("..."));
  assert.ok(isHallucination(Array(20).fill("de l'honneur").join(" ")));
  assert.ok(isHallucination("hmhmhmhmhmhmhmhmhmhmhmhmhmhmhm"));
  assert.ok(!isHallucination("Les sous-titres du film étaient en anglais."));
  assert.ok(!isHallucination("Merci beaucoup à tous pour votre présence aujourd'hui."));
  assert.ok(!isHallucination("Oui."));
});

test("isSilent / speechScore", () => {
  const silence = new Float32Array(SR * 30);
  assert.ok(isSilent(silence));
  assert.equal(speechScore(silence), 0);
  const tone = silence.map((_, i) => 0.3 * Math.sin(i / 10));
  assert.ok(!isSilent(tone));
  assert.equal(speechScore(tone), 1);
  const quiet = silence.map((_, i) => 0.0008 * Math.sin(i / 10)); // ≈ -65 dBFS : bruit de fond
  assert.ok(isSilent(quiet));
  const distant = silence.map((_, i) => 0.004 * Math.sin(i / 10)); // ≈ -51 dBFS : parole lointaine
  assert.ok(!isSilent(distant));
});

test("combineLanguageVotes: les fenêtres sûres d'elles l'emportent", () => {
  // Cas réel observé : fenêtre pleine de français (97 %) + fenêtres pauvres qui hésitent vers l'anglais.
  const r = combineLanguageVotes([
    [["fr", 0.97], ["en", 0.013]],
    [["en", 0.39], ["fr", 0.18]],
    [["en", 0.41], ["fr", 0.2]],
  ]);
  assert.equal(r[0][0], "fr");
  assert.ok(r[0][1] > 0.75, `confiance ${r[0][1]}`);
  const sum = r.reduce((a, [, p]) => a + p, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
});

test("pickLanguageWindows: fenêtres complètes et les plus fortes, seuil relatif", () => {
  const tone = (amp, n = SR * 30) => Float32Array.from({ length: n }, (_, i) => amp * Math.sin(i / 7));
  const windows = [
    { pcm: new Float32Array(SR * 30), full: true, id: "silence" },
    { pcm: tone(0.004), full: true, id: "faible" },      // ≈ -51 dBFS : enregistrement lointain
    { pcm: tone(0.008), full: true, id: "moyen" },
    { pcm: tone(0.3, SR * 10), full: false, id: "court-fort" },
  ];
  const picked = pickLanguageWindows(windows, 2).map((w) => w.id);
  assert.deepEqual(picked, ["moyen", "faible"]);
  // Seulement des fenêtres courtes : on les prend quand même.
  assert.deepEqual(pickLanguageWindows([windows[3]], 5).map((w) => w.id), ["court-fort"]);
  // Tout est silencieux : on garde la première fenêtre.
  assert.deepEqual(pickLanguageWindows([windows[0]], 5).map((w) => w.id), ["silence"]);
});

test("formatDuration", () => {
  assert.equal(formatDuration(45), "45 s");
  assert.equal(formatDuration(97), "1 min 37 s");
  assert.equal(formatDuration(12 * 60), "12 min");
  assert.equal(formatDuration(3 * 3600 + 5 * 60), "3 h 05");
  assert.equal(formatDuration(null), "—");
});

test("PcmBuffer: ajout, tranche, oubli du début", () => {
  const b = new PcmBuffer(1000);
  b.append(Float32Array.from({ length: 300 }, (_, i) => i));
  b.append(Float32Array.from({ length: SR * 200 }, () => 7)); // force l'agrandissement
  assert.equal(b.end, 1000 + 300 + SR * 200);
  assert.deepEqual([...b.slice(1100, 1103)], [100, 101, 102]);
  b.dropBefore(1250);
  assert.equal(b.start, 1250);
  assert.deepEqual([...b.slice(1250, 1252)], [250, 251]);
  assert.equal(b.slice(0, 1250).length, 0);
});

test("StreamingResampler: 48 kHz -> 16 kHz, longueur et amplitude", () => {
  const r = new StreamingResampler(48000, 16000);
  const x = Float32Array.from({ length: 48000 * 2 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / 48000));
  const parts = [];
  for (let i = 0; i < x.length; i += 777) parts.push(r.push(x.subarray(i, i + 777)));
  parts.push(r.flush());
  const y = parts.flatMap((p) => [...p]);
  assert.equal(y.length, 32000);
  const peak = Math.max(...y.slice(500, -500).map(Math.abs));
  assert.ok(Math.abs(peak - 0.5) < 0.005, `peak ${peak}`);
});

test("formats", () => {
  const t = { file: "a.mp3", language: "fr", duration: 3700, segments: [
    { start: 0, end: 2.5, text: " Bonjour. " }, { start: 3661.2, end: 3665.04, text: "Fin." }, { start: 3666, end: 3667, text: " " },
  ] };
  assert.equal(toTxt(t), "Bonjour.\nFin.\n");
  assert.match(toSrt(t), /^1\n00:00:00,000 --> 00:00:02,500\nBonjour\.\n\n2\n01:01:01,200 --> 01:01:05,040\nFin\.\n$/);
  assert.match(toVtt(t), /^WEBVTT\n\n00:00:00\.000 --> 00:00:02\.500\nBonjour\./);
  assert.equal(JSON.parse(toJson(t)).text, "Bonjour.\nFin.");
});

test("isRepeatLoop: 3e répétition identique supprimée", () => {
  const prev = [{ text: "Bonjour." }, { text: "No se puede ser muy bien," }, { text: "no se puede ser muy bien" }];
  assert.ok(isRepeatLoop("No se puede, ser muy bien.", prev));
  assert.ok(!isRepeatLoop("Autre chose.", prev));
  assert.ok(!isRepeatLoop("Oui.", [{ text: "Oui." }])); // deux occurrences restent permises
});
