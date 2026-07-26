// Policy inference in the browser, one onnxruntime-web session per precision.
//
// The ONNX graphs carry the observation normalization inside them, so the input is
// the raw 48-dim observation and the output is the tanh-squashed action. Nothing
// is normalized on the JavaScript side; doing so was the single most likely source
// of a silent correctness bug in this project and is avoided by construction.

import * as ort from '../vendor/ort/ort.wasm.bundle.min.mjs';

// Deliberately single-threaded and without SIMD auto-selection differences: the
// latency numbers this file measures have to describe one stated configuration,
// not whatever the browser happened to pick.
ort.env.wasm.numThreads = 1;
// Resolved against this module's own URL rather than the document, so the page
// works from any path. A relative string here is resolved by the runtime against
// its own bundle location and produces `vendor/ort/vendor/ort/…`.
ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;

export class Policy {
  constructor(session, name, inputName, outputName) {
    this.session = session;
    this.name = name;
    this.inputName = inputName;
    this.outputName = outputName;
    this.lastMs = 0;
    this.emaMs = 0;
  }

  static async load(url, name) {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new Policy(session, name, session.inputNames[0], session.outputNames[0]);
  }

  /** One forward pass. Returns a Float32Array of actions. */
  async act(obs) {
    const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
    const t0 = performance.now();
    const out = await this.session.run({ [this.inputName]: tensor });
    this.lastMs = performance.now() - t0;
    // Exponential moving average, so the HUD reads steadily rather than flickering.
    this.emaMs = this.emaMs === 0 ? this.lastMs : 0.9 * this.emaMs + 0.1 * this.lastMs;
    return out[this.outputName].data;
  }

  /**
   * Measured per-step latency: 100 warmup iterations discarded, then 1000 timed.
   *
   * Timed as **blocks**, not per call. Browsers deliberately coarsen
   * `performance.now()` — Chrome to 100 µs — and this policy is small enough that
   * a single forward pass falls below that. Timing each call individually
   * produced a median of exactly 0.000 ms with every sample landing on 0.0 or
   * 0.1: a measurement of the clock, not of the model. Blocks of `blockSize`
   * calls take long enough for the granularity to be negligible, and the spread
   * across blocks is a real quantity rather than timer quantisation.
   *
   * The per-call distribution is therefore *not* reported. It is not resolvable
   * on this timer, and inventing quantiles from quantised samples would be
   * fabricating the tail behaviour a control engineer would most want.
   */
  async benchmark(obs, { warmup = 100, iterations = 1000, blockSize = 50 } = {}) {
    for (let i = 0; i < warmup; i++) await this.act(obs);

    const blocks = [];
    const nBlocks = Math.max(1, Math.floor(iterations / blockSize));
    for (let b = 0; b < nBlocks; b++) {
      const t0 = performance.now();
      for (let i = 0; i < blockSize; i++) {
        const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
        await this.session.run({ [this.inputName]: tensor });
      }
      blocks.push((performance.now() - t0) / blockSize);
    }

    const sorted = [...blocks].sort((a, b) => a - b);
    const mean = blocks.reduce((a, b) => a + b, 0) / blocks.length;
    const quantile = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
    return {
      scheme: this.name,
      iterations: nBlocks * blockSize,
      warmup,
      block_size: blockSize,
      blocks: nBlocks,
      // Per-step, derived from block totals.
      mean_ms: mean,
      block_median_ms: quantile(0.5),
      block_p95_ms: quantile(0.95),
      block_min_ms: sorted[0],
      block_max_ms: sorted[sorted.length - 1],
      timer_resolution_note:
        'per-call latency is below the browser timer resolution; figures are ' +
        'block totals divided by block size, and the spread is across blocks',
    };
  }
}

/** Browser and CPU identity, recorded alongside any latency number. */
export function environmentReport() {
  return {
    user_agent: navigator.userAgent,
    hardware_concurrency: navigator.hardwareConcurrency ?? null,
    wasm_threads: ort.env.wasm.numThreads,
    // `deviceMemory` and CPU model are not exposed to pages; the user agent and
    // the machine disclosure in the README are what identify the host.
    platform: navigator.platform ?? null,
  };
}
