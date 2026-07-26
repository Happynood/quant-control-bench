// MuJoCo WASM wrapper: one independent simulation per precision variant.
//
// Everything that describes the model or the policy interface comes from
// assets/scene/scene.json, which `qcb export-scene` generates from the trained
// environment. Nothing here is hardcoded, because every one of those numbers is a
// silent-failure candidate: a wrong sensor address feeds the policy a different
// robot's velocity and it still walks, just worse.

// MuJoCo exposes some fields as typed-array views into WASM memory and others as
// embind vectors. The two have incompatible `set`: `TypedArray.set(array, offset)`
// copies a sequence, while `vector.set(index, value)` writes one element. Calling
// the wrong one is not a type error — it throws `offset is out of bounds`, or
// worse, silently writes the wrong thing. Views are detected first, explicitly.
const isView = (vec) => ArrayBuffer.isView(vec);

/** Read element `i` from a MuJoCo array. */
export function at(vec, i) {
  return isView(vec) ? vec[i] : vec.get(i);
}

/** Write element `i` of a MuJoCo array. */
export function put(vec, i, value) {
  if (isView(vec)) vec[i] = value;
  else vec.set(i, value);
}

export function readBlock(vec, adr, dim) {
  const out = new Float64Array(dim);
  for (let i = 0; i < dim; i++) out[i] = at(vec, adr + i);
  return out;
}

/**
 * A deterministic pseudo-random source for observation noise.
 *
 * The training environment draws its observation noise from JAX's counter-based
 * PRNG, which cannot be reproduced here. This is therefore *not* an attempt to
 * match Python bit-for-bit — with noise on, no browser rollout can. It exists so
 * the demo shows the deployed behaviour (the policy was trained with this noise)
 * while staying reproducible across reloads within the browser. The parity test
 * runs with `level = 0`, which removes the term entirely and makes the comparison
 * meaningful.
 */
export function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    // xorshift32: cheap, deterministic, adequate for a visual demo.
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

export class Sim {
  /**
   * @param {object} mujoco loaded WASM module
   * @param {object} scene parsed scene.json
   * @param {string} label which precision this instance represents
   */
  constructor(mujoco, scene, label) {
    this.mujoco = mujoco;
    this.scene = scene;
    this.label = label;
    this.iface = scene.policy_interface;

    this.model = mujoco.MjModel.loadFromXML(`/working/${scene.top_xml}`);
    this.#assertShape();
    this.#applyOverrides();
    this.data = new mujoco.MjData(this.model);

    this.nu = scene.nu;
    this.lastAction = new Float32Array(this.nu);
    this.prevAction = new Float32Array(this.nu);
    this.obs = new Float32Array(48);
    this.command = new Float32Array([1.0, 0.0, 0.0]);

    this.noiseLevel = this.iface.noise.level;
    this.rng = makeRng(0x9e3779b9);

    // Perturbations, driven by the UI. Model-level ones are applied to the
    // compiled model; loop-level ones act between policy and plant.
    this.actuatorDelay = 0;
    this.delayQueue = [];
    this.obsNoiseSigma = 0;

    this.reset();
  }

  #assertShape() {
    const m = this.model;
    const expected = [this.scene.nq, this.scene.nv, this.scene.nu];
    const actual = [m.nq, m.nv, m.nu];
    for (let i = 0; i < 3; i++) {
      if (expected[i] !== actual[i]) {
        throw new Error(
          `${this.label}: loaded model has nq/nv/nu ${actual.join('/')}, ` +
            `scene.json says ${expected.join('/')} — wrong scene or wrong assets`
        );
      }
    }
  }

  /**
   * Reapply the edits the training environment makes after parsing the XML.
   *
   * `Go1Env.__init__` sets the timestep, raises ccd_iterations, and rewrites joint
   * damping and every actuator gain and bias from its PD constants. None of it is
   * in the XML. Skipping this step produces a robot that walks but is not the one
   * that was benchmarked.
   */
  #applyOverrides() {
    const m = this.model;
    const s = this.scene;
    m.opt.timestep = s.timestep;
    m.opt.ccd_iterations = s.ccd_iterations;
    for (let i = 0; i < s.dof_damping.length; i++) put(m.dof_damping, i, s.dof_damping[i]);
    // Row widths come from scene.json. Both matrices are (nu, 10) in MuJoCo 3.x;
    // assuming 3 for biasprm — which its name and the mjNBIAS constant invite —
    // scatters each actuator's bias into another actuator's row.
    const gainStride = s.actuator_gainprm_stride;
    const biasStride = s.actuator_biasprm_stride;
    for (let i = 0; i < s.actuator_gainprm_col0.length; i++) {
      put(m.actuator_gainprm, i * gainStride + 0, s.actuator_gainprm_col0[i]);
      put(m.actuator_biasprm, i * biasStride + 1, s.actuator_biasprm_col1[i]);
    }
  }

  /**
   * Release the WASM-side model and data.
   *
   * Each Sim holds a compiled 10 MB model in the module's heap. Rebuilding the
   * variant list without this leaks one per rebuild, and a visitor toggling
   * checkboxes exhausts the heap in a couple of dozen clicks.
   */
  dispose() {
    for (const handle of [this.data, this.model]) {
      if (handle && typeof handle.delete === 'function' && !handle.isDeleted?.()) {
        handle.delete();
      }
    }
    this.data = null;
    this.model = null;
  }

  /** Scale torso mass; 1.0 restores the exported value. */
  setMassScale(scale) {
    const bodyId = 1; // first non-world body: the trunk
    if (this._baseMass === undefined) this._baseMass = at(this.model.body_mass, bodyId);
    put(this.model.body_mass, bodyId, this._baseMass * scale);
  }

  /** Scale the sliding friction coefficient of every geom. */
  setFrictionScale(scale) {
    const n = this.model.ngeom;
    if (this._baseFriction === undefined) {
      this._baseFriction = new Float64Array(n);
      for (let i = 0; i < n; i++) this._baseFriction[i] = at(this.model.geom_friction, i * 3);
    }
    for (let i = 0; i < n; i++) put(this.model.geom_friction, i * 3, this._baseFriction[i] * scale);
  }

  setActuatorDelay(steps) {
    this.actuatorDelay = Math.max(0, Math.round(steps));
    this.delayQueue = [];
  }

  setObsNoise(sigma) {
    this.obsNoiseSigma = sigma;
  }

  /** Apply an impulse to the torso's linear velocity, as the eval harness does. */
  push(impulseNs) {
    const mass = at(this.model.body_mass, 1);
    const dv = impulseNs / Math.max(mass, 1e-9);
    // Random direction in the horizontal plane plus a little vertical, unit-normalised.
    let v = [this.rng() * 2 - 1, this.rng() * 2 - 1, this.rng() * 2 - 1];
    const norm = Math.hypot(v[0], v[1], v[2]) || 1;
    for (let i = 0; i < 3; i++) {
      put(this.data.qvel, i, at(this.data.qvel, i) + (v[i] / norm) * dv);
    }
  }

  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
    // Start from the keyframe the environment uses (`home`), if present.
    if (this.model.nkey > 0) {
      for (let i = 0; i < this.scene.nq; i++) put(this.data.qpos, i, at(this.model.key_qpos, i));
    }
    this.lastAction.fill(0);
    this.prevAction.fill(0);
    this.delayQueue = [];
    this.totalReturn = 0;
    this.steps = 0;
    this.jitterSum = 0;
    this.jitterCount = 0;
    this.fallen = false;
    this.mujoco.mj_forward(this.model, this.data);
  }

  /**
   * Projected gravity: the world down-vector in the IMU site's frame.
   *
   * Matches `Go1Env.get_gravity` exactly —
   * `data.site_xmat[imu].T @ [0, 0, -1]`, which is the negated third row of that
   * row-major 3x3. It is **not** the `upvector` sensor; reading that instead
   * gives roughly the negation, in a frame that need not be the IMU's.
   *
   * These three numbers are the policy's orientation signal. Inverted, the policy
   * is told the robot is upside down and does the sensible thing: it stops
   * walking and tries to hold still. Measured with the wrong vector, the Go1
   * travelled 0.02 m in five seconds against a 1.0 m/s command.
   */
  gravity() {
    const base = this.iface.imu_site_id * 9 + 6;
    return [
      -at(this.data.site_xmat, base + 0),
      -at(this.data.site_xmat, base + 1),
      -at(this.data.site_xmat, base + 2),
    ];
  }

  #noise(scaleName) {
    const scale = this.iface.noise.scales[scaleName] ?? 0;
    return (2 * this.rng() - 1) * this.noiseLevel * scale;
  }

  /** Build the 48-dim observation exactly as `_get_obs` does. */
  observe() {
    const d = this.data;
    const s = this.iface.sensors;
    const o = this.obs;
    let k = 0;

    const linvel = readBlock(d.sensordata, s.local_linvel.adr, 3);
    for (let i = 0; i < 3; i++) o[k++] = linvel[i] + this.#noise('linvel');

    const gyro = readBlock(d.sensordata, s.gyro.adr, 3);
    for (let i = 0; i < 3; i++) o[k++] = gyro[i] + this.#noise('gyro');

    const g = this.gravity();
    for (let i = 0; i < 3; i++) o[k++] = g[i] + this.#noise('gravity');

    const q0 = this.iface.qpos_joint_start;
    for (let i = 0; i < 12; i++) {
      o[k++] = at(d.qpos, q0 + i) - this.iface.default_pose[i] + this.#noise('joint_pos');
    }

    const v0 = this.iface.qvel_joint_start;
    for (let i = 0; i < 12; i++) o[k++] = at(d.qvel, v0 + i) + this.#noise('joint_vel');

    for (let i = 0; i < 12; i++) o[k++] = this.lastAction[i];
    for (let i = 0; i < 3; i++) o[k++] = this.command[i];

    // The perturbation axis adds its own noise on top of the env's, matching
    // `perturbed_rollout`, which noises the observation the policy receives.
    if (this.obsNoiseSigma > 0) {
      for (let i = 0; i < o.length; i++) o[i] += this.obsNoiseSigma * gaussian(this.rng);
    }
    return o;
  }

  /**
   * Advance one control step: apply `action`, run `n_substeps` physics steps.
   * Returns the wall-clock spent inside physics, for the HUD.
   */
  step(action) {
    let applied = action;
    if (this.actuatorDelay > 0) {
      this.delayQueue.push(Float32Array.from(action));
      applied =
        this.delayQueue.length > this.actuatorDelay
          ? this.delayQueue.shift()
          : new Float32Array(this.nu);
    }

    if (this.steps > 0) {
      let delta = 0;
      for (let i = 0; i < this.nu; i++) delta += (applied[i] - this.prevAction[i]) ** 2;
      this.jitterSum += Math.sqrt(delta);
      this.jitterCount += 1;
    }
    this.prevAction = Float32Array.from(applied);
    this.lastAction = Float32Array.from(applied);

    const pose = this.iface.default_pose;
    const scale = this.iface.action_scale;
    for (let i = 0; i < this.nu; i++) put(this.data.ctrl, i, pose[i] + applied[i] * scale);

    const t0 = performance.now();
    for (let i = 0; i < this.scene.n_substeps; i++) {
      this.mujoco.mj_step(this.model, this.data);
    }
    const physicsMs = performance.now() - t0;

    this.steps += 1;
    // Uprightness, from the same projected-gravity vector the policy sees: its z
    // component is about -1 standing and turns positive once the robot is over.
    if (this.gravity()[2] > 0.0) this.fallen = true;

    // Command-tracking error: how far the achieved body velocity is from what the
    // joystick asked for. Exact, and directly meaningful to a viewer.
    //
    // The episodic *return* is deliberately not computed here. It is a sum of 16
    // environment-specific terms involving contact state, feet air time, swing
    // peaks and actuator forces; a JavaScript reimplementation would produce a
    // number that looks like the benchmark's return and is not it, which is worse
    // than not showing one. Measured returns live in the results panel, where
    // they come from the recorded runs.
    const linvel = readBlock(this.data.sensordata, this.iface.sensors.local_linvel.adr, 3);
    const gyro = readBlock(this.data.sensordata, this.iface.sensors.gyro.adr, 3);
    this.trackingError = Math.hypot(
      linvel[0] - this.command[0],
      linvel[1] - this.command[1],
      gyro[2] - this.command[2]
    );
    this.trackingSum = (this.trackingSum ?? 0) + this.trackingError;

    return physicsMs;
  }

  get meanTrackingError() {
    return this.steps > 0 ? (this.trackingSum ?? 0) / this.steps : 0;
  }

  get meanJitter() {
    return this.jitterCount > 0 ? this.jitterSum / this.jitterCount : 0;
  }
}

/** Box–Muller, for the observation-noise perturbation axis. */
export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
