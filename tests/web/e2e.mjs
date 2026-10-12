// Test de bout en bout de la page docs/ dans Chromium (Playwright), avec un vrai modèle Whisper.
//
// Les requêtes vers huggingface.co et cdn.jsdelivr.net sont servies depuis le disque :
//   MODEL_DIR  dossier contenant <org>/<modèle>/... (ex. un export de Xenova/whisper-tiny)
//   ORT_DIST   dossier dist/ du paquet npm onnxruntime-web (même version que transformers.js)
//   TJS_DIST   dossier dist/ du paquet npm @huggingface/transformers (version de docs/asr-worker.js)
//   AUDIO      fichier audio à transcrire
//   QUERY      paramètres d'URL (défaut : model=Xenova/whisper-tiny&dtype=q8&debug=1)
//   EXPECT     JSON optionnel {language, contains:[...], minSegments}
//   PW         chemin du module playwright ; CHROMIUM chemin de l'exécutable
//
// Exemple : MODEL_DIR=… ORT_DIST=… AUDIO=reunion.mp3 node tests/web/e2e.mjs
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { extname, join, resolve } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW || "playwright");
const ROOT = resolve(new URL("../..", import.meta.url).pathname);
const env = (k, d) => process.env[k] ?? d;
const MODEL_DIR = env("MODEL_DIR");
const ORT_DIST = env("ORT_DIST");
const AUDIO = env("AUDIO");
const QUERY = env("QUERY", "model=Xenova/whisper-tiny&dtype=q8&debug=1");
const EXPECT = JSON.parse(env("EXPECT", "{}"));
const TIMEOUT = Number(env("TIMEOUT_S", "900")) * 1000;

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".css": "text/css" };

// Petit serveur statique pour la racine du dépôt.
const server = createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (path.endsWith("/")) path += "index.html";
    const file = join(ROOT, path);
    const body = await readFile(file);
    res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://localhost:${server.address().port}`;

async function serveFile(route, file) {
  try {
    const info = await stat(file);
    const range = route.request().headers()["range"];
    const body = await readFile(file);
    const headers = { "access-control-allow-origin": "*", "content-type": TYPES[extname(file)] || "application/octet-stream", "accept-ranges": "bytes" };
    if (range) {
      const [, a, b] = /bytes=(\d+)-(\d*)/.exec(range);
      const start = Number(a), end = b ? Math.min(Number(b), info.size - 1) : info.size - 1;
      return route.fulfill({ status: 206, headers: { ...headers, "content-range": `bytes ${start}-${end}/${info.size}` }, body: body.subarray(start, end + 1) });
    }
    return route.fulfill({ status: 200, headers, body });
  } catch {
    return route.fulfill({ status: 404, headers: { "access-control-allow-origin": "*" }, body: "not found" });
  }
}

const browser = await chromium.launch({
  headless: true,
  executablePath: env("CHROMIUM", "/opt/pw-browsers/chromium"),
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"],
});
const context = await browser.newContext({ acceptDownloads: true });
const requests = [];
await context.route("https://huggingface.co/**", (route) => {
  const u = new URL(route.request().url());
  requests.push(u.pathname);
  const m = /^\/(.+?)\/resolve\/[^/]+\/(.+)$/.exec(decodeURIComponent(u.pathname));
  return m && MODEL_DIR ? serveFile(route, join(MODEL_DIR, m[1], m[2])) : route.fulfill({ status: 404, body: "" });
});
await context.route("https://cdn.jsdelivr.net/**", (route) => {
  const u = new URL(route.request().url());
  requests.push(u.pathname);
  const name = u.pathname.split("/").pop();
  if (u.pathname.includes("/@huggingface/transformers@")) return serveFile(route, join(env("TJS_DIST", ""), name));
  return ORT_DIST ? serveFile(route, join(ORT_DIST, name)) : route.fulfill({ status: 404, body: "" });
});

const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
page.on("worker", (w) => w.on("console", (m) => logs.push(`[worker ${m.type()}] ${m.text()}`)));

const t0 = Date.now();
await page.goto(`${base}/docs/?${QUERY}`);
await page.waitForSelector("body.ready", { timeout: 30000 });
const caps = await page.locator("#caps").innerText();
await page.setInputFiles("#file", AUDIO);
await page.waitForFunction(() => !document.getElementById("start").disabled, null, { timeout: 60000 });
const fileMeta = await page.locator("#fileMeta").innerText();
if (process.env.LANGUAGE) await page.selectOption("#language", process.env.LANGUAGE);
if (process.env.MULTILINGUAL) await page.check("#multilingual");
await page.click("#start");

let resumeInfo = null;
if (process.env.SCENARIO === "resume") {
  // Arrête après 3 fenêtres, rechoisit le fichier et reprend.
  await page.waitForFunction(() => /fenêtre 3 /.test(document.getElementById("debug").textContent), null, { timeout: TIMEOUT });
  await page.click("#cancel");
  await page.waitForFunction(() => !document.body.classList.contains("running"), null, { timeout: 120000 });
  const before = await page.locator("#live p").count();
  // Comme un vrai utilisateur : on reclique directement sur « Lancer » (sans rechoisir le fichier).
  await page.waitForFunction(() => !document.getElementById("resume").hidden, null, { timeout: 30000 });
  resumeInfo = { stageAfterCancel: await page.locator("#stage").innerText(), linesBefore: before, resumeText: await page.locator("#resumeText").innerText() };
  await page.click("#start");
  await page.waitForFunction(() => document.body.classList.contains("running"), null, { timeout: 60000 });
  await page.waitForFunction(() => document.querySelectorAll("#live p").length > 0, null, { timeout: 60000 });
  resumeInfo.firstLineAfterResume = await page.locator("#live p").first().innerText();
}

let langFixInfo = null;
if (process.env.SCENARIO === "langfix") {
  // Après la première fenêtre, l'utilisateur corrige la langue : la page doit tout relancer.
  await page.waitForFunction(() => /fenêtre 1 /.test(document.getElementById("debug").textContent), null, { timeout: TIMEOUT });
  const detected = await page.locator("#langInfo").innerText();
  await page.click("#langFix");
  await page.selectOption("#langSelect", process.env.FIX_LANGUAGE || "es");
  await page.click("#langApply");
  await page.waitForFunction(() => /Espagnol|Anglais|Allemand|Italien/.test(document.getElementById("langInfo").textContent), null, { timeout: TIMEOUT });
  langFixInfo = { detected, after: await page.locator("#langInfo").innerText() };
}

let outcome = "timeout";
let last = null;
const debugLines = [];
const deadline = Date.now() + TIMEOUT;
while (Date.now() < deadline) {
  const st = await page.evaluate(() => ({
    stage: document.getElementById("stage").textContent,
    pct: document.getElementById("pct").textContent,
    detail: document.getElementById("detail").textContent,
    debug: document.getElementById("debug").textContent,
    notice: document.getElementById("notice").hidden ? "" : document.getElementById("notice").textContent,
    done: !document.getElementById("result").hidden,
    running: document.body.classList.contains("running"),
  }));
  last = st;
  if (st.debug && debugLines.at(-1) !== st.debug) debugLines.push(st.debug);
  if (st.done) { outcome = "done"; break; }
  if (!st.running && /erreur|arrêtée/i.test(st.stage)) { outcome = `error: ${st.stage} ${st.notice}`; break; }
  await page.waitForTimeout(500);
}

const result = outcome === "done" ? await page.evaluate(() => ({
  meta: document.getElementById("resultMeta").textContent,
  text: document.getElementById("resultText").textContent,
  langInfo: document.getElementById("langInfo").textContent,
})) : null;

// Téléchargement SRT
let srt = null;
if (result) {
  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#dl-srt")]);
  srt = await readFile(await dl.path(), "utf8");
}

const report = {
  outcome, resumeInfo, langFixInfo, last: outcome === "done" ? undefined : last, seconds: Math.round((Date.now() - t0) / 1000), caps, fileMeta, result,
  srtHead: srt?.split("\n").slice(0, 8).join("\n"),
  gpu: debugLines.slice(-3), gpuSeries: debugLines.map((l) => /GPU (\d+) Mo/.exec(l)?.[1]).filter(Boolean).map(Number),
  requests: [...new Set(requests)],
  errors: logs.filter((l) => /error|pageerror/i.test(l)).slice(0, 20),
};
console.log(JSON.stringify(report, null, 2));

let ok = outcome === "done";
if (ok && EXPECT.language) ok &&= result.langInfo.toLowerCase().includes(EXPECT.language.toLowerCase());
if (ok && EXPECT.contains) ok &&= EXPECT.contains.every((w) => result.text.toLowerCase().includes(w.toLowerCase()));
if (process.env.SHOW_LOGS) console.log(logs.join("\n"));
await browser.close();
server.close();
process.exit(ok ? 0 : 1);
