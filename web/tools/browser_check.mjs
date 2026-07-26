// Drives the demo in a real browser: smoke check, physics parity, latency.
//
// Everything here runs the page's own code rather than reimplementing it, so a
// number reported by this script is a number the visitor's browser produces.
//
//   node web/tools/browser_check.mjs --mode smoke
//   node web/tools/browser_check.mjs --mode parity --steps 100 --out results/browser_parity.json
//   node web/tools/browser_check.mjs --mode latency --out results/browser_latency.json
//   node web/tools/browser_check.mjs --mode obs --out results/browser_obs.json
//   node web/tools/browser_check.mjs --mode race

import { chromium } from 'playwright';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, cur, i, arr) => {
    if (cur.startsWith('--')) acc.push([cur.slice(2), arr[i + 1]?.startsWith('--') ? true : arr[i + 1]]);
    return acc;
  }, [])
);

const URL = args.url ?? 'http://127.0.0.1:8765/index.html';
const MODE = args.mode ?? 'smoke';
const OUT = args.out ?? null;

function save(payload) {
  if (!OUT) {
    console.log(JSON.stringify(payload, null, 2));
    return;
  }
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify(payload, null, 2) + '\n');
  console.log(`wrote ${OUT}`);
}

const browser = await chromium.launch({
  executablePath:
    process.env.CHROMIUM ??
    `${process.env.HOME}/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome`,
  args: ['--no-sandbox', '--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text());
});
page.on('pageerror', (e) => errors.push(String(e)));

await page.goto(URL, { waitUntil: 'load' });
await page.waitForFunction(() => window.QCB !== undefined, null, { timeout: 180_000 });

const version = await page.evaluate(() => ({
  user_agent: navigator.userAgent,
  mujoco_wasm_version: window.QCB.state.mujoco.mj_versionString(),
  hardware_concurrency: navigator.hardwareConcurrency ?? null,
  scene: window.QCB.state.scene.env,
  nq: window.QCB.state.scene.nq,
  n_substeps: window.QCB.state.scene.n_substeps,
  variants: Object.keys(window.QCB.state.variants.variants),
}));

if (MODE === 'smoke') {
  // Step the loaded variants a little and confirm the simulation moves and the
  // policy produces finite actions.
  const result = await page.evaluate(async () => {
    const { entries } = window.QCB.state;
    const at = window.QCB.at;
    const before = entries.map((e) => Array.from({ length: 3 }, (_, i) => at(e.sim.data.qpos, i)));
    let finite = true;
    for (let s = 0; s < 50; s++) {
      for (const e of entries) {
        const a = await e.policy.act(e.sim.observe());
        if (![...a].every(Number.isFinite)) finite = false;
        e.sim.step(a);
      }
    }
    const after = entries.map((e) => Array.from({ length: 3 }, (_, i) => at(e.sim.data.qpos, i)));
    return {
      schemes: entries.map((e) => e.scheme),
      moved: entries.map((_, i) => Math.hypot(...after[i].map((v, j) => v - before[i][j]))),
      steps: entries.map((e) => e.sim.steps),
      fallen: entries.map((e) => e.sim.fallen),
      actions_finite: finite,
    };
  });
  save({ mode: 'smoke', version, result, console_errors: errors });
  if (!result.actions_finite) process.exitCode = 1;
  if (result.moved.some((d) => !(d > 1e-4))) {
    console.error('a robot did not move: the simulation is not stepping');
    process.exitCode = 1;
  }
}

if (MODE === 'parity') {
  // Deterministic open-loop trajectory: the *actions* come from a fixed script
  // supplied by Python, so this isolates the physics engines from the policy.
  // Observation noise is off (level 0) because the environment draws it from
  // JAX's PRNG, which no browser can reproduce.
  const steps = Number(args.steps ?? 100);
  const trajectory = await page.evaluate(async (nSteps) => {
    const { Sim } = window.QCB;
    const { mujoco, scene } = window.QCB.state;
    const sim = new Sim(mujoco, scene, 'parity');
    sim.noiseLevel = 0;
    sim.reset();

    // A fixed, policy-independent action sequence. Deterministic and identical
    // to what the Python side replays.
    const action = new Float32Array(sim.nu);
    const out = [];
    for (let t = 0; t < nSteps; t++) {
      for (let i = 0; i < sim.nu; i++) action[i] = 0.3 * Math.sin(0.05 * t + i);
      sim.step(action);
      const q = [];
      for (let i = 0; i < scene.nq; i++) q.push(window.QCB.at(sim.data.qpos, i));
      out.push(q);
    }
    return out;
  }, steps);
  save({ mode: 'parity', version, steps, qpos: trajectory, console_errors: errors });
}

if (MODE === 'obs') {
  // The observation the policy is fed, at a state Python can reproduce exactly.
  // Physics parity says nothing about this: a rollout with scripted actions never
  // calls `observe()`. The gravity vector was read from the wrong source for a
  // while and only showed up as a robot that would not walk.
  const out = await page.evaluate(() => {
    const { Sim, at } = window.QCB;
    const { mujoco, scene } = window.QCB.state;
    const sim = new Sim(mujoco, scene, 'obs');
    sim.noiseLevel = 0;
    sim.reset();

    const snapshot = (label) => ({
      label,
      qpos: Array.from({ length: scene.nq }, (_, i) => at(sim.data.qpos, i)),
      qvel: Array.from({ length: scene.nv }, (_, i) => at(sim.data.qvel, i)),
      gravity: sim.gravity(),
      obs: Array.from(sim.observe()),
    });

    const states = [snapshot('keyframe')];
    // A second state with the body moved and rotated, so an observation that is
    // only correct at the symmetric rest pose cannot pass.
    const action = new Float32Array(sim.nu);
    for (let t = 0; t < 20; t++) {
      for (let i = 0; i < sim.nu; i++) action[i] = 0.4 * Math.sin(0.11 * t + i);
      sim.step(action);
    }
    states.push(snapshot('after-20-steps'));
    return { command: Array.from(sim.command), last_action: Array.from(sim.lastAction), states };
  });
  save({ mode: 'obs', version, ...out, console_errors: errors });
}

if (MODE === 'race') {
  // Rapid selection changes used to interleave rebuilds and mix cards from one
  // selection with policies from another.
  const result = await page.evaluate(async () => {
    const boxes = [...document.querySelectorAll('input[name=scheme]')];
    const pick = (names) => {
      boxes.forEach((b) => {
        b.checked = names.includes(b.value);
      });
      document.getElementById('schemes').dispatchEvent(new Event('change', { bubbles: true }));
    };
    pick(['fp32']);
    pick(['fp32', 'fp16', 'int8-tensor']);
    pick(['ternary']);
    pick(['int8-channel', 'int4-group32']);
    const wanted = ['fp32', 'int4-channel'];
    pick(wanted);
    await new Promise((r) => setTimeout(r, 6000));
    return {
      wanted,
      entries: window.QCB.state.entries.map((e) => e.scheme),
      cards: [...document.querySelectorAll('#grid .card')].map((c) => c.dataset.scheme),
      policies: window.QCB.state.entries.map((e) => e.policy.name),
    };
  });
  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  const ok =
    same(result.entries, result.wanted) &&
    same(result.cards, result.wanted) &&
    same(result.policies, result.wanted);
  save({ mode: 'race', version, result, ok, console_errors: errors });
  if (!ok) {
    console.error('selection race: loaded variants do not match the final selection');
    process.exitCode = 1;
  }
}

if (MODE === 'latency') {
  const report = await page.evaluate(async () => await window.QCB.runLatencyBenchmark());
  save({ mode: 'latency', version, ...report, console_errors: errors });
}

if (errors.length) {
  console.error(`console errors:\n  ${errors.join('\n  ')}`);
  process.exitCode = 1;
}

await browser.close();
