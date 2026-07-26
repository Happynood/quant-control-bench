// Record the demo to a GIF for the README.
//
// Drives the page's own simulation and screenshots the viewport grid, so the
// animation is a recording of the real thing rather than an illustration of it.
// Frames are assembled by ffmpeg through a generated palette, which keeps the
// robot's edges from banding at GIF's 256 colours.
//
//   node web/tools/record_gif.mjs --out ../docs/assets/demo.gif

import { chromium } from 'playwright';
import { mkdirSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, cur, i, arr) => {
    if (cur.startsWith('--')) acc.push([cur.slice(2), arr[i + 1]]);
    return acc;
  }, [])
);

const URL = args.url ?? 'http://127.0.0.1:8765/index.html';
const OUT = resolve(args.out ?? 'demo.gif');
const SCHEMES = (args.schemes ?? 'fp32,int4-channel,ternary').split(',');
const FRAMES = Number(args.frames ?? 130);
const STEPS_PER_FRAME = Number(args.stepsPerFrame ?? 2);
const FPS = Number(args.fps ?? 25);
// Frame at which every robot takes the same impulse. This is the benchmark's own
// push axis, not a staged effect: at 8 N*s the measured fp32 success rate is 88%
// and ternary's is 0.
const PUSH_AT = args.pushAt === undefined ? -1 : Number(args.pushAt);
const PUSH_NS = Number(args.push ?? 8);
const TMP = resolve('.gif-frames');

rmSync(TMP, { recursive: true, force: true });
mkdirSync(TMP, { recursive: true });
mkdirSync(dirname(OUT), { recursive: true });

const browser = await chromium.launch({
  executablePath:
    process.env.CHROMIUM ??
    `${process.env.HOME}/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome`,
  args: ['--no-sandbox', '--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars'],
});
const page = await browser.newPage({ viewport: { width: 1180, height: 460 }, deviceScaleFactor: 2 });
await page.goto(URL, { waitUntil: 'load' });
await page.waitForFunction(() => window.QCB !== undefined, null, { timeout: 180_000 });

await page.evaluate(async (schemes) => {
  document.querySelectorAll('input[name=scheme]').forEach((b) => {
    b.checked = schemes.includes(b.value);
  });
  document.getElementById('schemes').dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 4000));
  // Fallen robots stay frozen at the moment they went over. The trained model
  // carries collision geometry on the feet only — a speed optimisation from
  // training, not a demo shortcut — so a torso that keeps integrating after a
  // fall sinks through the floor and reads as a rendering glitch rather than a
  // fall. Freezing shows the tip, which is the informative frame.
  document.getElementById('freeze').checked = true;
}, SCHEMES);

const grid = page.locator('#grid');
for (let f = 0; f < FRAMES; f++) {
  if (f === PUSH_AT) {
    await page.evaluate((ns) => {
      for (const e of window.QCB.state.entries) e.sim.push(ns);
    }, PUSH_NS);
  }
  await page.evaluate(async (steps) => {
    for (let s = 0; s < steps; s++) {
      const freeze = document.getElementById('freeze').checked;
      await Promise.all(
        window.QCB.state.entries.map(async (e) => {
          if (freeze && e.sim.fallen) return;
          const a = await e.policy.act(e.sim.observe());
          e.sim.step(a);
        })
      );
    }
    for (const e of window.QCB.state.entries) {
      e.viewport.sync();
      e.viewport.draw();
      window.QCB.updateHud(e);
    }
  }, STEPS_PER_FRAME);
  await grid.screenshot({ path: `${TMP}/f${String(f).padStart(4, '0')}.png` });
}

const summary = await page.evaluate(() =>
  window.QCB.state.entries.map((e) => ({
    scheme: e.scheme,
    steps: e.sim.steps,
    fallen: e.sim.fallen,
    tracking: e.sim.meanTrackingError,
  }))
);
await browser.close();

const palette = `${TMP}/palette.png`;
const filters = 'fps=' + FPS + ',scale=980:-1:flags=lanczos';
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS),
  '-i', `${TMP}/f%04d.png`, '-vf', `${filters},palettegen=stats_mode=diff`, palette]);
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS),
  '-i', `${TMP}/f%04d.png`, '-i', palette,
  '-lavfi', `${filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3`, OUT]);
rmSync(TMP, { recursive: true, force: true });

console.log(JSON.stringify({ out: OUT, frames: FRAMES, fps: FPS, schemes: SCHEMES, summary }, null, 1));
