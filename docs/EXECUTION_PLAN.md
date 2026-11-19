# External Force Estimation with an EKF, No RPM: Execution Plan

A standalone mini project for the course assignment *"Mini Project: Implementing State Estimation Filters"*
(`resources/MiniProject_FilterImplementation2 (1).pdf`). The design comes from `resources/plan.pdf`.

**Filter + application (the unique combination):** Extended Kalman Filter × estimating the external force on a
quadrotor from motor commands (PWM), battery voltage, IMU and RTK GNSS, without rotor-speed (RPM) sensing. Thrust,
thrust gain and motor lag are estimated online, so they don't corrupt the force estimate.

**Platform:** Python for the filter and evaluation. Isaac Sim 5.1 + Pegasus Simulator + PX4 SITL for the realistic
simulation, with PX4 data taken over the uXRCE-DDS link (ROS 2 Jazzy) and from PX4's flight logs (ULog).

**Scope rule.** Everything this project creates lives in `External-Force-Estimation/`. It uses the standalone
installs already on this PC as they are:

| Tool | Path | Used how |
|---|---|---|
| Isaac Sim 5.1 | `~/isaacsim` | run scripts with `~/isaacsim/python.sh <script>` |
| Pegasus Simulator | `~/Documents/Repos/PegasusSimulator` (registered as an Isaac extension folder) | imported by our scripts; we subclass its classes, we don't edit them |
| PX4-Autopilot | `~/Documents/Repos/PX4-Autopilot` (v1.18.0-rc1, `build/px4_sitl_raptor` built) | PX4 SITL in lockstep with Pegasus; parameters set at launch |
| ROS 2 Jazzy | `/opt/ros/jazzy` | DDS side: XRCE agent, offboard node, rosbag2 |

castor-main and the MARL project are **not** used or changed. What was learned from reading them (motor-lag
modelling, sensor/force ordering in Pegasus, DDS setup, cable-load truth) is written into this plan as our own code.

Order of work: **implement → test → evaluate properly → report (later).**

---

## Status (2026-11-18)

| Phase | State | Where |
|---|---|---|
| P0 setup + data-path validation | done: all three gate checks pass (motor signal 1.0000, frames, clock +4 ms) | `docs/FINDINGS.md` |
| P1 Tier 0 sim + filter core | done: 19 tests green (Jacobians, frames, truth models, known-answer, gate lock-out) | `efe/`, `tests/` |
| P2 Pegasus extensions | done: lockstep multirotor, voltage-lagged thrust, battery, IMU bias, force injector, truth log | `sim/` |
| P3 bench identification | done: k_b and u0 within 0.01 %, tau 44 vs 40 ms (lag on speed, not thrust), drag within 1 % | `efe/tools/bench.py` |
| P4 flights + pipeline | done: XRCE-DDS + rosbag2 + ULog ingest; DDS == ULog sample for sample | `tools/run_tier1*.sh`, `efe/io/ingest.py` |
| P5 filters on Tier 1 | done; EKF-C (tilt error) added after the PX4 attitude finding | `config/filters/` |
| P6 evaluation | done for both tiers: matrix (5 scenarios x 6 filters x seeds 1-4), 10 experiments, 50-run replay MC; Tier 1 from 25 Isaac/PX4 flights | `results/RESULTS.md` |
| P7 live node | not started (needs decision D1) | |
| P8 report | after evaluation | |

The repository layout below was the plan; the code lives in the `efe` package (`efe/estimation`, `efe/truth`,
`efe/pysim`, `efe/io`, `efe/evaluation`, `efe/tools`) with the Isaac side in `sim/`. See `README.md`.

---

## 1. How the assignment's criteria are covered

| Assignment requirement / criterion | Where this plan covers it |
|---|---|
| Implement KF / EKF / PF for an application | EKF (main), plus a linear KF baseline: §4, P1, P5 |
| Mathematical derivations | process and measurement models, Jacobians, observability: §4 (goes into the report's Methodology) |
| Code explanation | one module per concept, short and testable: §3 |
| Simulations or real data | two tiers: pure-Python sim, and Isaac + Pegasus + PX4 SITL: §2 |
| Visualisations | the figure list: §6.7 |
| Experiment with parameters and scenarios | scenario set and robustness sweeps: §6.4–6.6 |
| Correctness | Jacobian tests, NEES/NIS consistency, ground-truth comparison: §6.3 |
| Depth of understanding | observability of k and λ, KF vs EKF under battery sag, bias vs force separability |
| Effectiveness of the filter | accuracy targets vs baselines: §6.5 |
| Creativity / innovation | no-RPM thrust from PWM + voltage physics; online k_b and λ; accel-bias state; realistic PX4 pipeline |

---

## 2. Architecture

```
┌──────────── Isaac Sim 5.1 + Pegasus (our script: sim/efe_sim.py, Iris quadrotor) ────────────┐
│ EfeMultirotor (our subclass of Pegasus' Multirotor)                                           │
│   VoltageLaggedThrustCurve   ω_cmd,i = (1000·u_i + 100)·V/V₀ ; ω̇_i = (ω_cmd,i − ω_i)/τ_true   │
│                              T_i = k_f·ω_i²                                                   │
│   BatteryModel               V = OCV(SoC) − R₀·I(ΣT) ; SoC −= I·dt/Q                          │
│   ForceInjector              scheduled world-frame force → body FLU, re-applied every step    │
│   TruthLogger                every physics step: t_usec, p, v, q, specific force, F_ext,      │
│                              T_true, ω_i, u_i, V_true, SoC, k_eff                              │
└───────────────┬───────────────────────────────────────────────────────────────────────────────┘
                │ MAVLink HIL, TCP 4560, lockstep (PX4 clock = Pegasus time_usec)
┌───────────────▼──────────────┐
│ PX4 SITL (none_iris)          │──► ULog (all topics, high rate) ──► runs/<id>/px4.ulg
│ stock MC position controller  │
└───────────────┬──────────────┘
                │ uXRCE-DDS, UDP 8888
┌───────────────▼──────────────┐
│ MicroXRCEAgent + ROS 2 Jazzy  │──► rosbag2 ──► runs/<id>/bag/
│ offboard_node: flies the      │    (sensor_combined, vehicle_attitude, vehicle_odometry,
│ scenario (TrajectorySetpoint) │     vehicle_gps_position, vehicle_local_position, battery_status)
└──────────────────────────────┘
                                   ▼
        tools/ingest.py: ULog + bag + truth → one aligned Parquet per run (PX4 µs clock)
                                   ▼
        estimation/ (KF, EKF-A, EKF-B, EKF-B+bias, MO) → eval/ (metrics, NEES/NIS, figures)
```

**Two simulation tiers, one filter code base:**

- **Tier 0, pure Python** (`sim/pysim/`): same truth model (motor lag, voltage-scaled ESC, battery, forces, sensors)
  with a simple controller. Used to build and unit-test the filter, check Jacobians, and run large Monte Carlo and
  parameter sweeps in minutes.
- **Tier 1, Isaac + Pegasus + PX4 SITL over DDS**: the real flight controller, real topic rates and timing. This is
  the main evidence in the report. Runs at real time (~2–3 min per flight).

The filter runs **offline** on recorded runs (reproducible, fast to iterate). A live ROS 2 node is an optional extra.

---

## 3. Repository layout

```
External-Force-Estimation/
├── resources/                      assignment brief, design (plan.pdf)
├── docs/EXECUTION_PLAN.md          this file
├── docs/FINDINGS.md                measured facts from P0 (rates, signs, clock, motor order)
├── pyproject.toml                  package `efe`: numpy, scipy, pandas, pyarrow, pyulog, matplotlib, pyyaml, pytest
├── config/
│   ├── vehicle_iris.yaml           mass, drag d, k_f, u₀, τ_true, V₀ (truth side)
│   ├── battery_4s.yaml             OCV(SoC) table, R₀, Q (small, so V sags 20–30 % in ~2 min)
│   ├── px4_params.yaml             PX4 parameters for every run
│   ├── filters/*.yaml              KF / EKF-A / EKF-B / EKF-B+bias: Q, R, P₀, priors
│   └── scenarios/*.yaml            flight profile, force schedule, seeds
├── sim/
│   ├── efe_sim.py                  Tier 1 launcher (Isaac + Pegasus + PX4)
│   ├── pegasus_ext/
│   │   ├── multirotor.py           EfeMultirotor: sensors → PX4 → rotor forces in one step
│   │   ├── voltage_thrust.py       VoltageLaggedThrustCurve(QuadraticThrustCurve)
│   │   ├── battery.py              BatteryModel (numpy only, shared with Tier 0)
│   │   ├── force_injector.py       schedule → per-step force
│   │   └── truth_logger.py         buffered Parquet writer
│   ├── bench_id_sim.py             pinned-vehicle thrust sweep + step test
│   └── pysim/                      Tier 0: dynamics.py, controller.py, sensors.py, run.py
├── ros2_ws/src/
│   ├── px4_msgs/                   generated from PX4-Autopilot/msg (must match v1.18)
│   └── efe_ros/                    offboard_node.py, record.sh, [online_ekf_node.py]
├── estimation/
│   ├── frames.py                   NED/FRD ↔ ENU/FLU, quaternions (unit-tested)
│   ├── models.py                   f(x,u), h(x), analytic Jacobians
│   ├── ekf.py                      one EKF class with flags → EKF-A, EKF-B, EKF-B+bias
│   ├── kf_baseline.py              fixed k, τ (linear KF)
│   ├── momentum_observer.py        non-probabilistic baseline (Tomić-style)
│   └── attitude.py                 R at IMU timestamps (slerp / gyro propagation)
├── tools/
│   ├── px4_sitl.sh                 start PX4 SITL with config/px4_params.yaml
│   ├── ingest.py                   ULog / bag / truth → aligned Parquet
│   ├── bench_fit.py                fit u₀, k_b, V₀, τ, drag d
│   └── replay_mc.py                resample sensor noise on a Tier 1 truth trajectory
├── eval/
│   ├── metrics.py                  RMSE, bias, rise time, detection delay, NEES, NIS, ANEES
│   ├── plots.py                    the figures
│   └── run_matrix.py               scenario × filter × seed → tables + figures
├── tests/                          pytest: frames, Jacobians, filter on Tier 0, ingest/alignment
├── results/                        tables/ and figures/ (generated)
└── runs/                           (gitignored) runs/<scenario>/<seed>/{truth.parquet, px4.ulg, bag/, meta.yaml}
```

---

## 4. Model (the derivation section of the report)

### 4.1 Facts about this setup that shape the model

1. **Thrust curve has an offset.** Pegasus turns PX4's motor command u ∈ [0,1] into a rotor-speed reference
   `ω = 1000·u + 100` rad/s, and thrust per rotor is `k_f·ω²`. So thrust ∝ `(u + 0.1)²`, not `(1−a)u + a·u²`. The filter
   uses `σ(u) = Σ_i ((u_i + u₀)/(1 + u₀))²` with u₀ fitted on the bench (truth 0.1).
2. **Pegasus has no motor lag and no voltage effect.** Its rotors follow the command instantly. Our
   `VoltageLaggedThrustCurve` adds both, otherwise there is nothing for the EKF to learn.
3. **Pegasus applies forces late.** Its `Multirotor` applies the previous rotor command, then exchanges with PX4, then
   reads sensors: several physics steps from a measurement to the force it causes. `EfeMultirotor` reorders this
   (sensors → PX4 → forces) so the latency is one step, as on a real flight controller.
4. **Which PWM signal.** Pegasus receives `HIL_ACTUATOR_CONTROLS`, which is PX4's output after the thrust-model
   factor. With `THR_MDL_FAC = 0`, `actuator_motors` equals what the simulated ESC gets. Verified in P0.
5. **Accelerometer bias is large.** Pegasus' IMU draws a turn-on accelerometer bias with σ = 0.196 m/s². On a 1.5 kg
   drone that is ≈ 0.3 N of fake force per axis, more than the accuracy target, and PX4's `sensor_combined` is not
   bias-corrected. So the filter estimates b_a (15 states). Running with b_a off is an ablation.
6. **Drag is in the body frame.** Pegasus: `F_drag = −diag(d)·Rᵀv` (body), d = [0.5, 0.3, 0.0]. The filter uses the
   same form.
7. **Rates.** Over DDS, `vehicle_attitude` is limited to 50 Hz and `battery_status` to 1 Hz; `sensor_combined` is
   unlimited. The filter runs at the IMU rate and gets R between attitude samples by slerp or gyro propagation. V
   comes from our battery model (PX4 SITL simulates its own, unrelated battery).
8. **`actuator_motors` is not published over DDS** in this PX4 (only `/fmu/in/actuator_motors`), but it is in the
   ULog. ULog is therefore the source for u_i; DDS carries the other topics (see D1 in §9).

### 4.2 Frames and state

World NED, body FRD, R = body → world (from PX4's quaternion), e₃ = [0, 0, 1], gravity +g along +z.

State (15): `x = [p(3), v(3), F(3), T, k_b, λ, b_a(3)]`

| State | Meaning |
|---|---|
| F | external force, world frame: the target |
| T | actual collective thrust |
| k_b | residual thrust gain, what voltage doesn't explain (≈ constant) |
| λ = 1/τ | motor bandwidth |
| b_a | accelerometer bias, body frame |
| p, v | needed to propagate the dynamics and to use GNSS |

Inputs: u_i (motor commands), V (low-passed battery voltage, 1–5 Hz), R. Constants: m, g, d, V₀, u₀.

### 4.3 Process model (at the IMU rate, dt from PX4 timestamps)

```
s(V) = (V/V₀)²
σ(u) = Σ_i ((u_i + u₀)/(1 + u₀))²

p⁺   = p + v·dt
v⁺   = v + [ (−T·R·e₃ + F − R·d·Rᵀ·v)/m + g·e₃ ]·dt       thrust along −z_body (FRD)
F⁺   = F   + w_F        random walk, q_F sets the force bandwidth
T⁺   = T + λ·(k_b·s(V)·σ(u) − T)·dt                        ← nonlinear (products λ·k_b, λ·T)
k_b⁺ = k_b + w_k        very small
λ⁺   = λ   + w_λ        tiny
b_a⁺ = b_a + w_b        bias random walk
```

Non-trivial Jacobian entries:
`∂v⁺/∂v = I − R·d·Rᵀ·dt/m`, `∂v⁺/∂F = I·dt/m`, `∂v⁺/∂T = −R·e₃·dt/m`,
`∂T⁺/∂T = 1 − λ·dt`, `∂T⁺/∂k_b = λ·s·σ·dt`, `∂T⁺/∂λ = (k_b·s·σ − T)·dt`.

### 4.4 Measurements

| Sensor | Topic | Rate | h(x) | Non-zero H blocks |
|---|---|---|---|---|
| Accelerometer (specific force, FRD) | `sensor_combined` | IMU rate | `[−T·e₃ + Rᵀ(F − R·d·Rᵀ·v)]/m + b_a` | v: `−d·Rᵀ/m`, F: `Rᵀ/m`, T: `−e₃/m`, b_a: `I` |
| GNSS position, velocity | `vehicle_gps_position` → local NED | 10 Hz | `[p; v]` | `[I₆ 0]` |
| k_b pseudo-measurement (optional) | bench prior | 1 Hz | `k_b` | regulariser for unexcited hover |

### 4.5 Observability (key discussion point)

- **k_b and λ are never measured.** Their H columns are zero. They are learned through the dynamics: they drive T, and
  T appears in the accelerometer. The correction reaches them through the cross-covariances P_{T,k_b} and P_{T,λ}.
- **k_b vs F_z:** in steady hover `acc_z ≈ (k_b·s·σ + F_z)/m`; one equation, two unknowns. They separate only when σ(u)
  changes (k_b scales the changing part, a constant F_z doesn't). Throttle excitation is required.
- **λ:** only visible in transients (the delay of acc_z after a command step). In steady hover its covariance stays flat.
- **b_a vs F:** the filter propagates v with the model (thrust + F), not with the accelerometer, so GNSS pins down F;
  b_a is what remains in the accelerometer residual. Slower than the F–accelerometer path.
- **Voltage input:** most of the gain drift is explained by `(V/V₀)²`, so k_b barely moves and converges faster.
- Shown in the results as σ_k_b and σ_λ shrinking during excitation and flat in hover.

### 4.6 Ground truth

F_ext,true = the injected force (world NED), logged every physics step. Drag is not part of F_ext because the filter
models it; the drag-error experiment shows how much leaks into F̂ when d is wrong.

---

## 5. Phases

Each phase ends with a **gate**: a check that must pass before the next phase starts. Days are working days.

### P0. Setup and data-path validation (days 1–2)

Settles the three biggest risks (signs, the motor-command signal, time alignment) on a plain hover before any
filter exists.

1. `git init`, skeleton above, `pyproject.toml`, `pytest`, `.gitignore` (`runs/`, `ros2_ws/build`, `ros2_ws/install`).
2. `ros2_ws`: `px4_msgs` generated from `PX4-Autopilot/msg` so it matches v1.18; `colcon build`.
3. XRCE agent: build Micro-XRCE-DDS-Agent (v2.4.x) into `third_party/` or use the snap; run `MicroXRCEAgent udp4 -p 8888`.
4. PX4: use the existing `px4_sitl_raptor` build with `MC_RAPTOR_ENABLE=0` (the stock controllers are in it), or
   build `px4_sitl_default` (D2). `tools/px4_sitl.sh` starts it from its own working dir under `runs/<id>/px4/` with
   `PX4_SIM_MODEL=none_iris` and these parameters (`config/px4_params.yaml`, passed as `PX4_PARAM_*`):
   `THR_MDL_FAC=0`, `COM_LOW_BAT_ACT=0`, `SIM_BAT_MIN_PCT=50`, `UXRCE_DDS_SYNCT=0`, `COM_RC_IN_MODE=4`, `NAV_DLL_ACT=0`,
   `NAV_RCL_ACT=0`, `COM_DISARM_PRFLT=-1`, `SDLOG_MODE=1`, `SDLOG_PROFILE` with the high-rate bit,
   `IMU_INTEG_RATE` = physics rate, `EKF2_HGT_REF=1` (GNSS height).
5. `sim/efe_sim.py` v0: Iris, Pegasus PX4 backend (`px4_autolaunch: False`), GNSS set to RTK-like noise (eph 2 cm,
   10 Hz), physics 250 Hz (try 400 Hz), headless option, real-time pacing, truth logger.
6. `efe_ros/offboard_node.py` v0: arm, offboard, hover at 3 m for 30 s, land.
7. Record ULog + rosbag2. Write `docs/FINDINGS.md`:
   - topic rates (`ros2 topic hz`, and in the ULog);
   - `actuator_motors[i]` vs the u_i Pegasus received: identical with `THR_MDL_FAC=0`;
   - motor order PX4 i ↔ Pegasus `rotor{i}` ↔ position (quad-X);
   - PX4 `timestamp` vs Pegasus `time_usec`: the offset, measured, not assumed;
   - signs: hover `sensor_combined.accel_z ≈ −g` (FRD);
   - the Iris mass and inertia read from the USD.

**Gate P0:** in one hover run, after alignment: (a) truth specific force in FRD matches `sensor_combined` within noise
+ bias; (b) `Σ k_f(1000u+100)²` from logged `actuator_motors` matches T_true within 1 % in steady state; (c) clocks align
within one physics step.

### P1. Tier 0 simulator and filter core (days 3–6)

1. `sim/pysim`: 6-DoF translational dynamics + kinematic attitude (commanded tilt with a first-order lag), per-motor
   lag on ω, voltage-scaled ESC, `BatteryModel`, `ForceInjector`, sensors with Pegasus' noise parameters, a PD/PID
   position controller producing u.
2. `estimation/frames.py` + tests.
3. `estimation/models.py` + `tests/test_jacobians.py`: analytic vs finite-difference Jacobians at random states.
4. `estimation/ekf.py`: predict/update, Joseph form, asynchronous updates by timestamp, innovation and NIS logging,
   χ² gating, flags: `estimate_k`, `estimate_lambda`, `use_voltage`, `estimate_bias`.
5. `kf_baseline.py` (fixed k, τ, no V: linear model → plain KF), `attitude.py`.

**Gate P1 (on Tier 0):** EKF-B+bias tracks a −5 N step with RMSE < 0.3 N after convergence; ANEES over 50 runs inside
the 95 % χ² band; the KF drifts > 1 N in F_z under a 25 % battery sag. If the KF does *not* drift, the sag scenario is
too weak; fix it before moving on.

### P2. Pegasus truth-model extensions (days 5–8, overlaps P1)

1. `EfeMultirotor(Multirotor)`: update order sensors → backend → forces.
2. `VoltageLaggedThrustCurve(QuadraticThrustCurve)`: scale each reference by V/V₀ (clip at max rotor speed), then a
   first-order lag per rotor (`keep = exp(−dt/τ)`). Truth-mismatch knobs, off by default: voltage exponent ≠ 2,
   per-motor k_f spread ±3 %.
3. `BatteryModel`: 4S OCV table, R₀ ≈ 20 mΩ, `I = c·ΣT^1.5`, small Q so V drops 20–30 % in ~120 s.
4. `ForceInjector`: step, sinusoid + bias, impulse, ramp. World NED → Isaac ENU → body FLU with the current attitude;
   applied on `/body` through `apply_force` **every physics step** (the force lasts one step only).
5. `TruthLogger`: physics callback, buffered, written to Parquet at the end.

**Gate P2:** hovering with no force, V sags ≥ 20 % in one flight and PX4 still holds altitude (its integrator absorbs
the sag, which is exactly why a fixed-gain KF will see a fake force). With a −5 N step, the logged F_ext matches the
schedule and the altitude dips and recovers.

### P3. Bench identification in simulation (days 8–9)

`sim/bench_id_sim.py`: the Iris pinned to the world (fixed joint), PX4 not involved, u driven directly.

1. Thrust sweep u = 0.1 … 1.0 at full battery: fit u₀, k_b; record V₀.
2. Repeat at low SoC: thrust falls with V (the plot that justifies the voltage input).
3. Step test u 0.4 → 0.7: fit τ from `T(t) = T₀ + ΔT(1 − e^(−t/τ))`.
4. Drag: free-flight segments with F = 0 at several speeds → least-squares fit of d.
5. Save to `config/priors.yaml`; give the filter perturbed values (k_b +5 %, τ +20 %), like a real bench-to-flight mismatch.

**Gate P3:** fitted values within 2 % (k_b, u₀) and 5 % (τ) of truth before perturbation.

### P4. Scenario flights and data pipeline (days 9–12)

1. `offboard_node.py` flies each scenario from YAML (`TrajectorySetpoint` at 50 Hz). The sim runs at real time so
   PX4's offboard timeout is not hit.
2. Scenarios:

| ID | Flight | Forces | Purpose |
|---|---|---|---|
| S1 main | take-off, ±0.5 m vertical steps every 5 s (10–35 s), hover, force events, figure-8 (90–120 s) | −5 N z step (45–65 s); 2 N 0.5 Hz lateral sinusoid + 1 N bias (65–85 s); 10 N × 0.1 s impulse (90 s) | everything, battery sag throughout |
| S2 hover-only | hover, no excitation | same forces | observability of k_b vs F_z and λ |
| S3 sag-only | hover + vertical steps | none | pure battery drift: the KF fake-force result |
| S4 aggressive | fast figure-8 | gust | drag/force separation under motion |
| S5 payload-like | hover, lateral moves | slowly swinging hanging-load force (pendulum model in the injector) | realistic load |

3. `tools/ingest.py`: ULog (pyulog) + bag (rosbag2_py) + truth → one Parquet per run on the PX4 µs clock; truth
   resampled onto IMU timestamps; GNSS lat/lon/alt → local NED. ULog is the source for u_i; DDS and ULog values are
   cross-checked on shared topics.
4. S1–S5 × 5 seeds (≈ 25 runs × ~3 min), scripted to run unattended.

**Gate P4:** every run ingests without gaps; DDS and ULog agree on shared samples.

### P5. Filters on Tier 1 data (days 12–15)

1. Run KF, EKF-A, EKF-B, EKF-B+bias, MO on S3 first, then S1. Debug order: accelerometer innovation mean ≈ 0 in hover
   (signs) → NIS → NEES against truth.
2. Tune Q/R once on seed 0, then **freeze**; evaluate on seeds 1–4 only. Frozen values in `config/filters/`.
3. MO: momentum observer with fixed k, τ, gain set to match EKF-B's bandwidth.

**Gate P5:** EKF-B+bias meets the targets in §6.5 on S1, or the miss is understood and written down.

### P6. Evaluation (days 15–20)

As specified in §6. One command regenerates every table and figure:
`python -m eval.run_matrix --scenarios S1-S5 --filters all --seeds 1-4`.

### P7. Live ROS 2 node (optional, days 20–22)

`online_ekf_node.py`: subscribes to the DDS topics, publishes F̂ and its σ as `geometry_msgs/WrenchStamped`. Needs
`actuator_motors` on DDS (D1). Check: same output as the offline run on the same bag; compute time per step.

### P8. Report (after P6 is final)

Structure required by the brief: Introduction, Methodology, Implementation, Results, Discussion, Conclusion.

1. **Introduction:** the problem, why no RPM, the gap (PX4's EKF2 has no force estimate; Tomić/McKinnon assume RPM).
2. **Background:** KF/EKF theory, related work (Tomić, McKinnon, Tagliabue).
3. **Methodology:** §4 of this plan: models, Jacobians, voltage physics, observability.
4. **Implementation:** code walkthrough with snippets (models, EKF step, Pegasus extensions, pipeline).
5. **Results:** §6 figures and tables.
6. **Discussion:** what each component buys; limitations (R treated as noise-free, drag model, k_b/F_z coupling in
   hover, simulation-only validation); comparison with the momentum observer.
7. **Conclusion and future work:** attitude in the state, torque estimation, real-flight validation with a known
   hanging mass, live onboard node.

Every number and figure comes from `results/`, generated by code.

---

## 6. Evaluation specification

### 6.1 Filters compared (all on the same recorded data)

| ID | k | λ | Voltage | b_a | Isolates |
|---|---|---|---|---|---|
| KF | fixed (bench) | fixed (bench) | no | no | naive linear baseline |
| EKF-A | estimated | estimated | no | no | online gain without physics |
| EKF-B | k_b estimated | estimated | yes | no | the plan.pdf design |
| **EKF-B+bias** | k_b estimated | estimated | yes | estimated | proposed filter for PX4 data |
| MO | fixed | fixed | no | n/a | non-probabilistic baseline |

### 6.2 Accuracy (against the Pegasus truth log)

| Metric | Applies to | Definition |
|---|---|---|
| RMSE | F_x, F_y, F_z | whole flight and per phase (hover / step / gust / figure-8) |
| Steady-state bias | F_z | mean error in hover windows, plotted against SoC |
| Rise time 10→90 % | F_z step | from the step instant |
| Detection delay | impulse | time until ‖F̂‖ exceeds 3σ of its hover noise |
| Noise σ | F̂ in hover | estimate jitter |
| Parameter error | k_b·s(V) vs k_eff; λ vs 1/τ; b_a vs Pegasus' bias | final error, time to within 5 % |
| Compute | per filter step | mean / p99 µs |

### 6.3 Consistency (correctness of the filter)

- **NEES** per block (F; T, k_b, λ; b_a; full), averaged over runs; ANEES against the χ² 95 % band.
- **NIS** for accelerometer and GNSS updates (needs no truth, so it also works on real logs later).
- Error with ±3σ envelopes for F_z, k_b·s(V), λ, b_a,z.

### 6.4 Monte Carlo

- Tier 1: 5 seeds per scenario (noise seeds and force magnitudes ±20 %).
- Replay MC (`tools/replay_mc.py`): one Tier 1 truth trajectory, IMU/GNSS/V noise resampled 50×, all filters rerun:
  enough runs for NEES statistics.
- Tier 0: the parameter sweeps below (hundreds of runs); a subset confirmed on Tier 1.
- Every metric reported as mean ± std.

### 6.5 Success targets (Iris, ~1.5 kg)

- Lateral F RMSE < 0.3 N, vertical < 0.5 N (S1, excluding 1 s after the impulse).
- F_z bias under ≥ 20 % battery sag: < 0.2 N for EKF-B+bias; expected > 1 N for the KF.
- Step rise time < 0.5 s; impulse detection < 0.1 s.
- ANEES inside the 95 % χ² band; NIS inside its band ≥ 90 % of the time.

### 6.6 Parameter and robustness experiments

| Experiment | Variation | Tier |
|---|---|---|
| Observability | S1 vs S2: σ_k_b, σ_λ over time | 1 |
| Bandwidth vs noise | q_F over 3 decades → RMSE-vs-noise curve | 0, 1 replay |
| Prior error | k_b ±10 %, τ ±50 % → convergence time | 0 |
| Model error | wrong u₀, mass ±10 %, drag ±50 %, truth voltage exponent 1.8 | 0, spot-check 1 |
| Accelerometer bias | b_a state on/off; Pegasus bias on/zeroed | 1 |
| GNSS | 10 → 5 → 1 Hz; 5 s dropout during the gust | 1 replay |
| Voltage | noise ×5; no low-pass; 1 Hz (PX4 `battery_status` over DDS) | 1 replay |
| Attitude source | 50 Hz + slerp vs gyro propagation vs full-rate ULog | 1 |

### 6.7 Figures

1. F̂ vs truth, 3 axes, all filters (S1).
2. **F_z bias vs time as the battery drains: KF vs EKF-A vs EKF-B(+bias).** The headline result.
3. k_b·s(V) and λ vs truth with ±3σ bands.
4. NEES and NIS with χ² bands.
5. Observability: σ_k_b and σ_λ, excited vs hover-only.
6. q_F sweep: RMSE vs noise trade-off.
7. Bench ID: thrust curve at full and low battery; step-response fit.
8. Hanging-load scenario: F̂ vs truth.
9. Robustness table.

---

## 7. Timeline (~4 weeks to final evaluation, report after)

| Days | Phase | Output |
|---|---|---|
| 1–2 | P0 | validated hover run; `docs/FINDINGS.md` |
| 3–6 | P1 | Tier 0 sim, EKF core, tests green |
| 5–8 | P2 | Pegasus extensions; battery sag in flight |
| 8–9 | P3 | priors from simulated bench ID |
| 9–12 | P4 | all scenario runs recorded and ingested |
| 12–15 | P5 | filters on Tier 1, frozen tuning |
| 15–20 | P6 | metrics, Monte Carlo, robustness, figures |
| 20–22 | P7 | (optional) live node |
| then | P8 | report |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| NED/FRD vs ENU/FLU sign errors | one `frames.py`, unit-tested; Gate P0 compares truth and PX4 specific force directly |
| Motor-command definition (THR_MDL_FAC, order, scaling) | `THR_MDL_FAC=0`; u logged on both sides and compared in P0 |
| Time alignment | PX4 µs everywhere; Pegasus `time_usec` logged; offset measured |
| Accelerometer bias read as force | b_a state; ablation with bias zeroed |
| Isaac slower than real time → offboard timeout | headless, one drone; check the real-time factor; drop physics to 250 Hz |
| PX4 battery failsafe mid-run | `COM_LOW_BAT_ACT=0`, `SIM_BAT_MIN_PCT=50` |
| Truth model = filter model (results too good) | truth mismatch knobs (voltage exponent, per-motor spread, lag on ω not T) |
| k_b / F_z unobservable in hover | excitation in S1; optional k_b pseudo-measurement; S2 shown honestly |
| Duplicate topic with another student | confirm the "EKF × external force estimation" combination with the instructor early |

---

## 9. Decisions

- **D1. `actuator_motors` over DDS.** This PX4 doesn't publish it; the ULog has it, so P0–P6 need no PX4 change. The
  optional live node (P7) needs one line added to `src/modules/uxrce_dds_client/dds_topics.yaml` and a rebuild.
  Decide at P7: add it, or skip P7.
- **D2. PX4 build.** Default: the existing `px4_sitl_raptor` build with `MC_RAPTOR_ENABLE=0`. Alternative:
  `make px4_sitl_default` in `PX4-Autopilot` for a stock build.
