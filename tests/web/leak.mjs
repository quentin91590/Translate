// Lance tests/web/leak.html dans Chromium (WebGPU logiciel SwiftShader) et affiche l'évolution
// de la mémoire GPU. Mêmes variables que e2e.mjs (MODEL_DIR, ORT_DIST, TJS_DIST, PW) + PCM
// (fichier float32 16 kHz mono), N, NOLEAKFIX=1 pour désactiver le correctif.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { extname, join, resolve } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW || "playwright");
const ROOT = resolve(new URL("../..", import.meta.url).pathname);
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = path === "/pcm.f32" ? process.env.PCM : join(ROOT, path);
  try { res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" }).end(await readFile(file)); }
  catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const serve = async (route, file) => {
  try { await route.fulfill({ status: 200, headers: { "access-control-allow-origin": "*", "content-type": TYPES[extname(file)] || "application/octet-stream" }, body: await readFile(file) }); }
  catch { await route.fulfill({ status: 404, headers: { "access-control-allow-origin": "*" }, body: "" }); }
};
const browser = await chromium.launch({
  headless: true, executablePath: process.env.CHROMIUM || "/opt/pw-browsers/chromium",
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"],
});
const ctx = await browser.newContext();
await ctx.route("https://huggingface.co/**", (route) => {
  const m = /^\/(.+?)\/resolve\/[^/]+\/(.+)$/.exec(decodeURIComponent(new URL(route.request().url()).pathname));
  return serve(route, join(process.env.MODEL_DIR, m[1], m[2]));
});
await ctx.route("https://cdn.jsdelivr.net/**", (route) => {
  const u = new URL(route.request().url()), name = u.pathname.split("/").pop();
  return serve(route, join(u.pathname.includes("@huggingface/transformers@") ? process.env.TJS_DIST : process.env.ORT_DIST, name));
});
const page = await ctx.newPage();
const q = `model=${process.env.MODEL || "Xenova/whisper-tiny"}&dtype=${process.env.DTYPE || "q8"}&n=${process.env.N || 4}&pcm=/pcm.f32${process.env.NOLEAKFIX ? "&noleakfix" : ""}`;
await page.goto(`http://localhost:${server.address().port}/tests/web/leak.html?${q}`);
await page.waitForFunction(() => window.report?.done, null, { timeout: Number(process.env.TIMEOUT_S || 3600) * 1000, polling: 2000 });
const report = await page.evaluate(() => window.report);
console.log(JSON.stringify({ leakFix: !process.env.NOLEAKFIX, mb: (b) => b, afterLoadMB: Math.round(report.afterLoad / 1048576), series: report.series.map((s) => ({ gpuMB: +(s.gpu / 1048576).toFixed(1), ms: s.ms, text: s.text })), error: report.error }, null, 1));
await browser.close();
server.close();
