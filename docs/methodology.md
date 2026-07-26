# Methodology

Every measurement in this benchmark, and the reasoning behind the choices that produced it. Where a result is negative, or a metric had to be redefined to mean anything, that is recorded here too.

## Reference machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB |
| Driver | 595.71.05 |
| Python | 3.12 |
| JAX / jaxlib | 0.11.0 (CUDA 12 plugin) |
| MuJoCo / MJX | 3.10.0 |
| MuJoCo Playground | 0.2.0 |
| Brax | 0.14.2 |
| ONNX / onnxruntime | 1.22.0 / 1.27.0 |

Exact pins live in `uv.lock`; every result JSON additionally embeds the versions that were live when it was produced.

## Physics backend

All environments are loaded through `quant_control_bench.envs.load_env`, which forces the MJX **JAX** implementation (`impl="jax"`).

Two reasons:

1. **Determinism.** The divergence-horizon metric `T_div` compares a quantized policy against the fp32 baseline from an identical initial state under an identical RNG stream. This is only well defined if the rollout is reproducible bit-for-bit given a fixed key. The JAX implementation is.

2. **The default does not work with the pinned stack.** MuJoCo Playground 0.2.0 together with mujoco-mjx 3.10.0 resolves the default `impl` to the Warp backend whenever `warp-lang` is importable. `warp-lang` arrives as a transitive dependency of Playground, but the separate `mujoco_warp` package does not, and `mjx.put_model` then fails with `AttributeError: type object 'int' has no attribute 'WARP'`. Pinning the backend in one place removes both the crash and the possibility of two runs silently using different physics.

## Numerical precision of the baseline

The fp32 baseline has to actually be fp32. On this machine it was not, by default.

JAX's default matmul precision for float32 inputs on an Ampere GPU truncates the mantissa to bfloat16 width. Measured here on a 256x512 @ 512x256 product, as maximum relative error against a float64 reference:

| setting | max relative error |
|---|---|
| JAX default | 3.075e-04 |
| explicit `bfloat16` | 3.075e-04 |
| explicit `highest` / `float32` | 4.913e-07 |
| NumPy float32 (CPU) | 4.913e-07 |

The default is bit-identical to requesting bfloat16.

This was found through a failing check rather than by reading documentation. The trainer verifies that the policy weights it extracts reproduce Brax's own forward pass; the check failed at 7.7e-03 on a trained CartpoleBalance policy. A layer-by-layer comparison showed the weights and the observation normalization agreeing to 2e-07, which located the discrepancy in the matmul itself rather than in the extraction. Forcing `jax_default_matmul_precision="highest"` brought the same comparison to **1.19e-07**.

Why it matters here specifically:

1. **Every delta is measured against the fp32 baseline.** A baseline whose matmuls carry an 8-bit mantissa is already quantized, so the measured cost of int8 would be taken from the wrong origin.
2. **The induced action error is ~3e-04** — the same order of magnitude as the error weight quantization is expected to introduce. Left in place it would confound the exact quantity this benchmark exists to measure.
3. **The browser computes in real float32.** `onnxruntime-web` does not use tensor cores, so a policy evaluated in Python at the default precision is not the policy that runs in the demo.

Every JAX entry point in the package therefore calls `quant_control_bench.precision.enforce_fp32_matmul` before touching a device, and a test asserts the measured relative error stays below 1e-05.

## Reproducibility of the simulator

Every comparison in this benchmark is a difference between two nearly identical policies. That only means anything if the simulator returns the same trajectory when nothing changed. On Tier 1 it did not.

Two `Go1JoystickFlatTerrain` rollouts of the same policy, same seed, same environment object, in the same process:

| | |
|---|---|
| Episodes with identical return | **0 of 100** |
| Largest disagreement | 1.03 of return, against a mean of 31.5 (~3%) |
| `CartpoleBalance` under the same test | bit-identical |

The policy was not the variable. The two bundles compared identical weight for weight after a reload, and their open-loop action error over the 10 000-state replay buffer was exactly zero. What differs is the GPU reduction order: contact-rich MJX dynamics accumulate through non-deterministic atomics, and a locomotion task amplifies a few-ULP difference within a handful of steps. Cartpole has no contacts to accumulate, which is why it was unaffected.

This invalidated three things at once:

1. **The `T_div` self-consistency requirement.** Rolling the baseline against itself is supposed to give the full horizon. It gave **6 steps out of 1000**.
2. **The paired comparison.** Two precisions are meant to differ only by their weights. With the simulator itself varying, the pairing was broken and the bootstrap interval was partly measuring simulator noise.
3. **The resolution floor.** fp32 evaluated against itself scored **+0.109% [−0.039%, +0.377%]** — a scheme is identical to the baseline and the harness still reported a non-zero delta with an interval. Any true effect below roughly 0.4% would have been indistinguishable from noise, and reported as if it were real.

`--xla_gpu_deterministic_ops=true` removes it entirely: 0 of 100 episodes differ, maximum delta exactly zero. It costs about **38% wall clock** on Tier 1 (45.6 s against 33.0 s per 100-episode rollout). The flag is set at package import in `quant_control_bench/__init__.py`, because XLA reads its flags when the backend initializes and a lazy call would be too late to have any effect. Every result manifest records `deterministic_ops` so a JSON file states which regime produced it.

**Why this was not caught earlier.** The self-consistency test existed from Phase 4 and passed continuously — on `CartpoleBalance`, which satisfies it without the flag. A test that can only run on the environment that cannot fail it proves nothing about the one that can. Both the self-consistency assertion and a plain rollout-reproducibility assertion now also run on Tier 1, marked `slow`.

## Environment scale

The VRAM protocol starts at `num_envs=4096` and halves on OOM. It did not need to halve. Measured on the reference machine with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, `jax.vmap`-ed reset plus 100 jitted steps under a zero action:

| env | num_envs | observation | action | control dt | env-steps/s | peak VRAM |
|---|---|---|---|---|---|---|
| `CartpoleBalance` | 4096 | 5 | 1 | 0.0100 s | 5,775,339 | 405 MiB |
| `Go1JoystickFlatTerrain` | 4096 | dict: `state` (48), `privileged_state` (123) | 12 | 0.0200 s | 30,550 | 423 MiB |

Because Tier 1 fits, the reduced-scale fallback tier is **not** in use and no result in this repository comes from it.

## Training budget

`num_timesteps` in Brax PPO is not a stopping point. Brax divides the request into whole epochs and rounds up:

```
env_step_per_training_step = batch_size * unroll_length
                             * num_minibatches * action_repeat
steps_per_epoch = ceil(num_timesteps
                       / (num_evals_after_init
                          * env_step_per_training_step
                          * max(num_resets_per_eval, 1)))
```

so the smallest run is one training step per epoch. For Playground's tuned CartpoleBalance config that floor is

```
1024 * 30 * 32 * 10 * (10 - 1) = 88,473,600 environment steps
```

A request for 10M therefore executes 8.8x more than asked, and the shipped "60M" tuned budget also really runs 88.47M. Training records store the request and the executed total as separate fields, and `budget_was_reduced` is evaluated against what actually ran.

Measured on the reference machine for CartpoleBalance at `num_envs=2048`: one epoch of 9,830,400 steps takes about 163 s, so the full tuned run is roughly 28 minutes — comfortably inside the 6-hour cap, and no reduction is needed for Tier 0.

## Observation layout

`Go1JoystickFlatTerrain` returns a dict observation for an asymmetric actor-critic: the policy consumes `state` (48 dims), while the critic additionally sees `privileged_state` (123 dims), which contains quantities a real robot cannot measure. Only the policy half is exported to ONNX. The key is resolved in exactly one place (`envs/registry.py`) so that an export cannot silently feed the critic observation to the policy network.

## Determinism of evaluation

Not yet exercised — the evaluation harness is not built. The intended contract: deterministic policy (mean action, no sampling), fixed evaluation seeds declared as constants in the config rather than derived from the clock, and a fixed environment RNG stream. Configs are rejected at load time if they name a quantization scheme or perturbation axis outside the declared sets.

## Statistics

Every headline delta carries a percentile-method bootstrap 95% confidence interval over 10 000 resamples; failure rates carry Wilson score intervals.

**Resampling is paired.** A quantized policy and the fp32 baseline are evaluated from identical initial states under an identical RNG stream, so their episode returns are matched pairs. Resampling them independently would inflate the interval with between-episode difficulty variance that cancels exactly in the difference. Relative deltas recompute the ratio inside each resample rather than propagating the absolute interval, because a ratio of means is not a linear function of the pair.

**Wilson, not the normal approximation, for failure rates.** Failure rates in this benchmark are routinely exactly 0. The normal approximation returns a zero-width interval there and claims certainty from a finite sample.

## Divergence horizon in practice

The headline definition fixes `epsilon = 0.1` on the per-DOF-normalized state distance. On a stabilization task that single threshold saturates. Measured on Tier 0, the baseline's per-DOF spread is small — the pole-angle standard deviation is 0.0026 rad — so `epsilon = 0.1` in normalized units corresponds to 2.6e-4 rad, below the action noise floor of int4 quantization. Every scheme coarser than int8 then reports a horizon of 1 step and the metric carries no information.

A ladder of thresholds is therefore evaluated in the same rollout pass, at no extra rollout cost. The headline definition is unchanged; the ladder simply makes the ordering visible. Median first-crossing step on Tier 0, 100 starts, 1000-step horizon:

| scheme | eps=0.01 | eps=0.1 | eps=1.0 | eps=10 |
|---|---|---|---|---|
| `fp32` | 1000 | 1000 | 1000 | 1000 |
| `fp16` | 1000 | 1000 | 1000 | 1000 |
| `int8-channel` | 2 | 1000 | 1000 | 1000 |
| `int8-tensor` | 1 | 31 | 1000 | 1000 |
| `int8-act` | 1 | 1 | 162 | 1000 |
| `mixed-head-fp16` | 1 | 1 | 37 | 1000 |
| `int4-channel` | 1 | 1 | 12 | 1000 |
| `ternary` | 1 | 1 | 1 | 15 |

At `epsilon = 1.0` the ordering is monotone in precision, which is what the metric is for. Tier 1's locomotion gait has a far larger per-DOF spread, so the saturation is expected to be a Tier 0 artifact rather than a permanent property.

The self-consistency requirement holds: rolling the baseline against itself never diverges at any threshold, over the full horizon.

## Defining success for the robustness frontier

`P50` needs a success criterion, and termination is not usable as one: CartpoleBalance never terminates, so a policy that has completely stopped balancing still reports a 0% failure rate. Success is a return threshold, set at half of the **fp32 baseline's** unperturbed return and shared by every scheme.

Scoring each policy against its own unperturbed return was tried first and is wrong. It ranked `ternary` the *most* robust scheme on the observation-noise axis, at `P50 = 0.600` against fp32's 0.275 — because ternary's own unperturbed return had already collapsed to 269 against fp32's 999, so it was graded against a bar less than a third as high. Against the shared reference it ranks last, at 0.167. A policy that has already failed must not be able to pass by failing consistently.

Where the success rate never crosses 50% inside the swept grid, `P50` is reported as the last swept magnitude and flagged as censored, never extrapolated.

### Surviving the episode is part of succeeding

The return threshold alone is not sufficient, and Tier 1 is where that showed. The push axis applies its impulse at a uniformly random step of a 1000-step episode, so an episode pushed at step 900 has already banked about 90% of its return and clears a 50% bar whether or not it then falls over. Roughly half of all episodes therefore succeed by construction.

Measured on the fp32 Go1 policy, 100 episodes per point:

| push (N·s) | success rate | episodes that fell | mean return |
|---|---|---|---|
| 0 | 100% | 0% | 31.493 |
| 16 | 70% | 63% | 21.211 |
| 32 | 58% | 88% | 17.505 |
| 64 | 55% | 91% | 16.814 |
| 128 | 54% | 94% | 16.421 |
| 256 | 54% | 88% | 16.421 |
| 512 | 45% | 80% | `nan` |

The success rate flattens just above 50% and never crosses it, while the fraction of episodes that actually fall climbs to 94%. `P50` was undefined on this axis for a reason that had nothing to do with robustness, and widening the grid does not help — at 256 N·s the success rate is still 54%.

Success is therefore three conditions: the episode ran to the horizon without terminating, its return was finite, and that return was at least half the fp32 baseline's. A robot lying on its back is not a success regardless of what it earned beforehand.

On `CartpoleBalance` this is exactly the previous criterion, and that is a measurement rather than an assumption: across every point of the Tier 0 observation-noise sweep the fraction of episodes that terminated early is 0%, so the survival clause is vacuous there and the returns are all finite. The balance task has no termination condition to trip.

### Grids are calibrated per environment, against the measured baseline

A magnitude grid is only meaningful for the robot it was chosen for. The first Tier 1 attempt reused grids picked for Cartpole and three of the five axes came out unusable: friction bottomed out at ×0.3 with the baseline still succeeding 89% of the time, and both mass and observation noise put their entire transition inside a single grid interval, so `P50` was an interpolation across one segment and could not separate the schemes.

The ranges that seem natural a priori do not perturb this policy at all. Measured on the fp32 Go1 baseline: 100% success at mass ×1.3 and 98% at friction ×0.5. MuJoCo Playground trains Go1 with its own domain randomization, so the trained policy's margin is wider than it looks, and the grids run past the suggested ranges until the baseline actually breaks. Each grid keeps points below the fp32 crossing as well, because a coarser scheme fails earlier and its crossing has to be resolved too.

Grids are therefore keyed by environment (`ENV_GRIDS`), and an environment without its own entry falls back to the Tier 1 calibration — a guess that should be checked against the baseline before its `P50` is quoted.

`actuator_delay` is a special case: the delay is `round(magnitude)` control steps, so the axis is integer-valued and fractional points are not distinct measurements. The baseline goes 100% → 70% → 3% over delays 0, 1, 2, which puts every scheme's crossing between 1 and 2 steps. That is the finest resolution the axis admits, and it limits how well this axis can separate precisions.

The `nan` at 512 N·s is the second half of the same fix. Extreme perturbations blow the integrator up; a non-finite return is a failed episode, not a missing one. Non-finite returns now count as failures, are excluded from the reported mean, and their fraction is recorded per grid point so an average is never read without knowing how many episodes went to infinity.

## A Tier 0 effect that did not replicate

On Tier 0, `int8-act` measures **4.0× more robust to observation noise than fp32** — `P50` 0.352 [0.332, 0.378] against 0.087 [0.083, 0.093], with non-overlapping intervals. (An earlier run of this axis used a 300-step horizon rather than the full 1000 and reported 1.214 against 0.272; the ratio survives the correction, the absolute magnitudes do not, because a shorter episode gives injected noise less time to accumulate into a failure. The numbers here are from the full horizon.) The mechanism was checked rather than assumed: activation quantization clamps activations to the calibrated range, and the clipped fraction rose monotonically with the injected noise (0.0000 at σ=0, 0.0212 at 0.1, 0.1376 at 0.4, 0.1910 at 0.8), so the quantizer really was acting as an input clamp. The result was recorded as axis-specific and flagged as requiring Tier 1 confirmation before it could be reported as robustness.

It does not replicate. On Go1, `int8-act` gives `P50` = 0.142 [0.138, 0.145] against fp32's 0.142 [0.138, 0.145] — identical to three decimal places. The Tier 0 effect was a property of `CartpoleBalance`, whose 5-dimensional observation and near-stationary state distribution make a calibrated clamp a meaningful filter. A 48-dimensional locomotion observation gives it nothing to clamp usefully.

This is why the smoke tier is not allowed to produce headline claims.

## Python ↔ browser parity

The demo runs the MuJoCo C engine compiled to WebAssembly; training and evaluation run MJX. The spec asks for the divergence to be measured and documented rather than assumed to be zero, and it is not zero.

Three things are held fixed so that only the engines differ: both sides build from the same exported scene bundle including its post-parse overrides, both replay the same fixed analytic action sequence rather than a policy, and observation noise is switched off. That last one is not optional — the environment draws its noise from JAX's counter-based PRNG, which JavaScript cannot reproduce, so with noise on no browser rollout could match a Python one however correct the physics.

Measured, 100 control steps (5 physics steps each), maximum absolute `qpos` difference:

| | |
|---|---|
| immediately after reset | **0.000e+00** |
| after 1 control step | 6.8e-04 |
| after 2 control steps | 2.1e-03 |
| after 10 | 4.2e-02 |
| after 100 | 8.8e-02 (peak over the run 5.1e-01) |

The zero at reset is the important number: the compiled models, the applied overrides and the initial keyframe agree exactly, so the divergence is purely the integrators and it compounds from there. Contact-rich dynamics amplify any difference quickly, which is the same mechanism the divergence-horizon metric measures between precisions.

**The engines are not the same version.** The published WASM package is MuJoCo **3.3.8**; the Python stack is **3.10.0**. Seven minor releases separate them, and a first-step difference of 6.8e-04 in a scene that starts in contact is consistent with that rather than with float round-off. The demo is therefore a faithful re-run of the same *model* and the same *policy*, not a bit-exact replay of a benchmarked trajectory, and no number quoted from the benchmark tables is produced in the browser.

Getting to a zero at reset took two fixes that would each have been silent:

- Playground's XML references meshes by paths that climb out of the installed package (`../../../../mujoco_menagerie/…`) with a `meshdir` to match. Python never follows them — `from_xml_string` resolves against an in-memory asset dict by basename — but MuJoCo in the browser reads a real filesystem. The exporter rewrites the references to bare filenames. Geometry is untouched, and the Python-side rebuild re-parses the rewritten files to prove it.
- `actuator_biasprm` is `(nu, 10)`, not `(nu, 3)` as its name and MuJoCo's `mjNBIAS` constant both suggest. Writing the PD bias at stride 3 scattered each actuator's bias into another actuator's row; the browser was 2.8e-02 from Python at the very first step. Both strides are now exported from the compiled model rather than assumed.

### Observation parity, and the bug physics parity could not see

Physics parity replays a scripted action sequence, so it never calls the code that builds the observation. That gap hid a real defect for the whole of phase 8: the demo fed the policy the wrong orientation vector and every physics check stayed green.

`Go1Env.get_gravity` is **not** a sensor read. It is

```python
data.site_xmat[imu_site_id].T @ [0, 0, -1]
```

— the world down-vector expressed in the IMU *site* frame, which equals the negated third row of that row-major 3×3. The exporter originally pointed the browser at the `upvector` sensor, the obvious guess, which gives roughly the negation in a frame that need not be the IMU's.

Those are 3 of the 48 observation entries, and they are the policy's primary orientation signal. Inverted, the policy is told the robot is upside down and does the sensible thing: it stops walking and tries to hold still. Measured, the Go1 travelled **0.02 m in five seconds** against a 1.0 m/s command with its command-tracking error sitting at exactly the command magnitude. After the fix, the same five seconds cover **4.77 m** and the tracking error falls to 0.06.

There is now an observation-parity mode that compares the browser's vectors against the environment's *own accessors* — not a second implementation of the same reading, which could share the misunderstanding:

| state | gravity | gyro | local linvel |
|---|---|---|---|
| keyframe (at rest) | **0.00e+00** | **0.00e+00** | **0.00e+00** |
| after 20 stepped control steps | 6.4e-03 | 3.2e-02 | 1.0e-02 |

The exact zero at rest is what confirms the wiring. The residual at the displaced state is *not* MJX's float32: the float64 C engine in Python 3.10.0 reproduces the MJX numbers, and both differ from the browser by the same amount. It is the same 3.3.8-versus-3.10.0 version gap already documented above, showing up in site kinematics rather than in the integrator.

## Browser inference latency

Measured in a real browser on the reference machine, 100 warmup iterations discarded then 1000 timed, single-threaded WASM.

**Timed in blocks, not per call.** Browsers coarsen `performance.now()` — Chrome to 100 µs — and one forward pass through this policy is faster than that. Timing each call individually returned a median of exactly 0.000 ms with every sample on 0.0 or 0.1: a measurement of the clock. Blocks of 50 calls put the total well above the granularity. The per-call distribution is deliberately **not** reported, because it is not resolvable and quantiles derived from quantised samples would fabricate exactly the tail behaviour a control engineer cares about.

Per-step means range from **0.033 to 0.053 ms** across the nine schemes, against a 20 ms control period — under 0.3% of the budget. The differences between schemes (~0.02 ms) are the same size as the block-to-block spread and should not be read as a ranking: quantization here is simulated, so every ONNX graph stores float32 weights and has the same shape. A packed-integer kernel would be a different measurement entirely.

## Known limitations

- **Single training seed (provisional).** The wall-clock budget caps any single training run at 6 hours. Whether three independently seeded policies fit inside that budget is not yet known; if only one is trained, precision effects cannot be fully separated from checkpoint luck, and that will be stated here and in the README rather than glossed over.
- **Browser physics differs from training physics.** The demo runs the MuJoCo C engine compiled to WebAssembly, while training uses MJX. The divergence between the two is to be measured and reported, not assumed to be zero.
