// Demo entry point: N precisions racing the same policy in the same physics.
//
// Every simulation shares an initial state, a command and a perturbation setting.
// The only difference between them is the precision of the weights driving them,
// which is the whole point of the page.

import loadMujoco from '../vendor/mujoco/mujoco_wasm.js';
import { Sim, at } from './sim.js';
import { Policy, environmentReport } from './policy.js';
import { Viewport } from './render.js';

const MODEL_BASE = window.QCB_MODEL_BASE ?? './assets/onnx';
const SCENE_BASE = './assets/scene';

const state = {
  mujoco: null,
  scene: null,
  variants: null,
  running: false,
  entries: [], // { scheme, sim, policy, viewport, hud }
  lastFrame: 0,
  fpsEma: 0,
  degraded: false,
};

async function loadScene(mujoco) {
  const scene = await (await fetch(`${SCENE_BASE}/scene.json`)).json();
  mujoco.FS.mkdir('/working');
  mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
  for (const name of scene.files) {
    const bytes = new Uint8Array(await (await fetch(`${SCENE_BASE}/${name}`)).arrayBuffer());
    mujoco.FS.writeFile(`/working/${name}`, bytes);
  }
  return scene;
}

// Built with DOM calls rather than an HTML string. The scheme ids come from this
// project's own registry, so interpolation would be safe here, but keeping the
// habit means a future value from anywhere else cannot become markup.
const HUD_FIELDS = [
  ['steps', 'steps survived', '0'],
  ['state', 'state', 'upright'],
  ['track', 'tracking error', '—'],
  ['jitter', 'action jitter', '—'],
  ['ms', 'inference', '—'],
];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function buildCard(scheme, bits) {
  const card = el('div', 'card');
  card.dataset.scheme = scheme;

  const head = el('div', 'card-head');
  head.append(el('span', 'scheme', scheme), el('span', 'bits', `${bits.toFixed(2)} bits/weight`));

  const list = el('dl', 'hud');
  for (const [key, label, initial] of HUD_FIELDS) {
    const row = el('div');
    const value = el('dd', null, initial);
    value.dataset.k = key;
    row.append(el('dt', null, label), value);
    list.append(row);
  }

  card.append(head, el('canvas'), list);
  return card;
}

/** Load the pieces that need awaiting. Touches no DOM and no shared state. */
async function loadVariant(scheme) {
  const info = state.variants.variants[scheme];
  const sim = new Sim(state.mujoco, state.scene, scheme);
  const policy = await Policy.load(`${MODEL_BASE}/${info.file}`, scheme);
  return { scheme, sim, policy, bits: info.bits_per_weight };
}

// Rebuilds are serialised by generation. Clicking several checkboxes in quick
// succession used to start overlapping rebuilds: each cleared the grid and each
// pushed into the same `state.entries`, so cards from one selection ended up
// alongside policies from another and the page showed variants nobody asked for.
// Loading now happens off to the side and is committed in one synchronous block,
// and a rebuild that has been superseded discards its work instead of publishing
// it.
let rebuildGeneration = 0;

async function rebuild() {
  const generation = ++rebuildGeneration;
  const wanted = [...document.querySelectorAll('input[name=scheme]:checked')].map((c) => c.value);

  const loaded = [];
  for (const scheme of wanted) {
    const variant = await loadVariant(scheme);
    if (generation !== rebuildGeneration) {
      // A newer selection is already in flight; drop what this one built.
      variant.sim.dispose();
      loaded.forEach((v) => v.sim.dispose());
      return;
    }
    loaded.push(variant);
  }
  if (generation !== rebuildGeneration) {
    loaded.forEach((v) => v.sim.dispose());
    return;
  }

  state.entries.forEach((e) => e.sim.dispose());

  const grid = document.getElementById('grid');
  const entries = loaded.map((v) => {
    const card = buildCard(v.scheme, v.bits);
    return {
      scheme: v.scheme,
      sim: v.sim,
      policy: v.policy,
      card,
      viewport: null,
    };
  });
  grid.replaceChildren(...entries.map((e) => e.card));

  // Viewports need their canvas attached to size correctly, so they are created
  // after the commit rather than before it.
  for (const entry of entries) {
    entry.viewport = new Viewport(entry.card.querySelector('canvas'), entry.sim, entry.scheme);
  }

  state.entries = entries;
  applyControls();
  resetAll();
  status(`${entries.length} variant(s) loaded: ${entries.map((e) => e.scheme).join(', ')}`);
}

function applyControls() {
  const mass = parseFloat(document.getElementById('mass').value);
  const friction = parseFloat(document.getElementById('friction').value);
  const delay = parseInt(document.getElementById('delay').value, 10);
  const noise = parseFloat(document.getElementById('noise').value);

  document.getElementById('mass-v').textContent = `${mass.toFixed(2)}x`;
  document.getElementById('friction-v').textContent = `${friction.toFixed(2)}x`;
  document.getElementById('delay-v').textContent = `${delay} step(s)`;
  document.getElementById('noise-v').textContent = noise.toFixed(3);

  const cmd = [
    parseFloat(document.getElementById('vx').value),
    0,
    parseFloat(document.getElementById('wz').value),
  ];
  document.getElementById('vx-v').textContent = `${cmd[0].toFixed(2)} m/s`;
  document.getElementById('wz-v').textContent = `${cmd[2].toFixed(2)} rad/s`;

  for (const e of state.entries) {
    e.sim.setMassScale(mass);
    e.sim.setFrictionScale(friction);
    e.sim.setActuatorDelay(delay);
    e.sim.setObsNoise(noise);
    e.sim.command.set(cmd);
  }
}

function resetAll() {
  for (const e of state.entries) {
    e.sim.reset();
    e.viewport.sync();
    e.viewport.draw();
  }
}

function pushAll() {
  const impulse = parseFloat(document.getElementById('push').value);
  for (const e of state.entries) e.sim.push(impulse);
  status(`pushed ${impulse.toFixed(1)} N·s`);
}

function status(text) {
  document.getElementById('status').textContent = text;
}

function updateHud(entry) {
  const q = (k) => entry.card.querySelector(`[data-k=${k}]`);
  q('steps').textContent = entry.sim.steps;
  q('state').textContent = entry.sim.fallen ? 'fallen' : 'upright';
  q('state').className = entry.sim.fallen ? 'bad' : 'ok';
  q('track').textContent = entry.sim.meanTrackingError.toFixed(3);
  q('jitter').textContent = entry.sim.meanJitter.toFixed(4);
  q('ms').textContent = `${entry.policy.emaMs.toFixed(2)} ms/step`;
}

async function frame(now) {
  if (!state.running) return;

  const dt = now - state.lastFrame;
  state.lastFrame = now;
  if (dt > 0) state.fpsEma = state.fpsEma === 0 ? 1000 / dt : 0.9 * state.fpsEma + 0.1 * (1000 / dt);

  // Mobile-degraded mode: if the page cannot hold 30 fps with the selected
  // variants, drop to two and say so rather than showing a slideshow.
  if (!state.degraded && state.fpsEma > 0 && state.fpsEma < 30 && state.entries.length > 2) {
    state.degraded = true;
    const boxes = [...document.querySelectorAll('input[name=scheme]')];
    boxes.forEach((b, i) => {
      b.checked = i < 2 ? b.checked || i === 0 : false;
    });
    if (!boxes.some((b) => b.checked)) boxes[0].checked = true;
    status(
      `below 30 fps with ${state.entries.length} robots — reduced to 2. ` +
        `Re-select variants to override.`
    );
    await rebuild();
    requestAnimationFrame(frame);
    return;
  }

  await Promise.all(
    state.entries.map(async (e) => {
      if (e.sim.fallen && document.getElementById('freeze').checked) return;
      const action = await e.policy.act(e.sim.observe());
      e.sim.step(action);
    })
  );

  for (const e of state.entries) {
    e.viewport.sync();
    e.viewport.draw();
    updateHud(e);
  }
  document.getElementById('fps').textContent = `${state.fpsEma.toFixed(0)} fps`;

  requestAnimationFrame(frame);
}

async function runLatencyBenchmark() {
  const wasRunning = state.running;
  state.running = false;
  status('measuring latency — 100 warmup + 1000 timed iterations per variant…');

  const obs = state.entries.length
    ? state.entries[0].sim.observe()
    : new Float32Array(state.variants.obs_dim);

  // Every loaded variant, not only the visible ones, so the recommender has a
  // latency number for each scheme it might select.
  const rows = [];
  for (const e of state.entries) {
    rows.push(await e.policy.benchmark(Float32Array.from(obs)));
  }

  const report = { environment: environmentReport(), measurements: rows };
  document.getElementById('latency').textContent = JSON.stringify(report, null, 2);
  status('latency measured — copy the JSON into results/browser_latency.json');

  state.running = wasRunning;
  if (wasRunning) {
    state.lastFrame = performance.now();
    requestAnimationFrame(frame);
  }
  return report;
}

async function main() {
  status('loading MuJoCo WebAssembly…');
  state.mujoco = await loadMujoco();
  state.scene = await loadScene(state.mujoco);
  state.variants = await (await fetch(`${MODEL_BASE}/variants.json`)).json();

  const chooser = document.getElementById('schemes');
  for (const scheme of Object.keys(state.variants.variants)) {
    const id = `scheme-${scheme}`;
    const wrap = el('label');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.name = 'scheme';
    box.id = id;
    box.value = scheme;
    // fp32 and int4-channel by default: the baseline and the one scheme measured
    // to lose robustness margin, which is the comparison the page is about.
    box.checked = scheme === 'fp32' || scheme === 'int4-channel';
    wrap.append(box, document.createTextNode(` ${scheme}`));
    chooser.appendChild(wrap);
  }

  for (const id of ['mass', 'friction', 'delay', 'noise', 'vx', 'wz']) {
    document.getElementById(id).addEventListener('input', applyControls);
  }
  document.getElementById('reset').addEventListener('click', resetAll);
  document.getElementById('push-btn').addEventListener('click', pushAll);
  document.getElementById('bench').addEventListener('click', runLatencyBenchmark);
  document.getElementById('play').addEventListener('click', () => {
    state.running = !state.running;
    document.getElementById('play').textContent = state.running ? 'pause' : 'play';
    if (state.running) {
      state.lastFrame = performance.now();
      requestAnimationFrame(frame);
    }
  });
  chooser.addEventListener('change', () => {
    state.degraded = false;
    rebuild();
  });
  window.addEventListener('resize', () => state.entries.forEach((e) => e.viewport.resize()));

  await rebuild();
  status('ready — press play');

  // Exposed for the headless parity and latency tests, which drive the page
  // directly rather than reimplementing any of it.
  window.QCB = { state, runLatencyBenchmark, rebuild, resetAll, updateHud, Sim, Policy, at };
}

main().catch((err) => {
  status(`failed: ${err.message}`);
  console.error(err);
});
