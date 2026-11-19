# Findings

Facts measured while building and validating the pipeline. Each one changed the code or the configuration, and each is
a point for the report's Methodology or Discussion. Dates: 2026-11-18.

## P0 gate: the data path (Tier 1: Isaac Sim 5.1 + Pegasus + PX4 v1.18 SITL, lockstep, 250 Hz)

| Check | Result | How |
|---|---|---|
| Motor command signal | `actuator_motors` = what the simulated ESC receives: thrust predicted from the logged commands / true thrust = **1.0000** in hover | `THR_MDL_FAC = 0`; Pegasus maps u -> omega = 1000 u + 100 |
| Frames and signs | accelerometer - true specific force = injected bias, to 1e-3 m/s^2 | `efe/io/ingest.py`, inverse-dynamics check |
| Clock | PX4 timestamps = simulator time **+ 4.0 ms** (one physics step), every run | cross-correlation of accel vs true specific force (`measure_clock_offset`) |
| Command latency | the command PX4 logs at t acts on the rotors from **t + 4 ms** (one step) | logged vs applied u (`motor_delay_measured_s`) |
| DDS path | every DDS sample of sensor_combined, vehicle_attitude, vehicle_gps_position equals the ULog sample with the same timestamp (**max diff 0.0**); DDS rates 250 / 50 / 10 Hz | `tools/bag_to_csv.py` + `ingest --check-bag` |
| Rates (ULog) | sensor_combined 250 Hz, vehicle_attitude 250 Hz, actuator_motors 250 Hz, vehicle_gps_position 10 Hz | `SDLOG_PROFILE = 27` |
| Real-time factor | 2.5x real time headless (RTX 5070 Ti), one drone | `efe_sim.py` |
| Unmodelled forces | none: m a - (thrust + gravity + drag) - injected force has mean 0.000 N in every flight phase | inverse dynamics on the truth log |

## Things that were wrong and are fixed

1. **Vehicle mass.** Pegasus' Iris is body 1.5 kg + four 0.03 kg rotor bodies = **1.6186 kg**. Reading only the body
   made the filter's thrust gain 7.5 % wrong. `efe_sim.py` now sums every articulation body.
2. **Pegasus applies forces late.** Stock `Multirotor` applies the previous command, then talks to PX4, then reads
   sensors. `EfeMultirotor` uses the lockstep order (sensors, PX4, forces): one step of latency, as in a real FC.
3. **Pegasus' accelerometer bias is never added** (the term is commented out in `imu.py`; its random walk also uses
   a uniform draw). `EfeIMU` adds a seeded Gauss-Markov bias (turn-on 0.196 m/s^2, as Pegasus' parameters intend).
4. **Pegasus' GPS noise is scaled by sqrt(dt)**, so its default densities give millimetre noise. Set to give RTK-level
   2 cm / 3 cm / 3 cm/s (`config/sim_tier1.yaml`).
5. **One-step command latency** (above) biased the motor bandwidth 27 % low and F_z by +0.5 to +0.8 N. The filter now
   takes a motor delay input (`motor_delay`, 4 ms in Tier 1, 0 in Tier 0); F_z RMSE 0.78 -> 0.12 N.
6. **Euler discretisation of the motor lag** read lambda about 5 % low at dt = 4 ms. The thrust row now uses the exact
   discretisation `T+ = T + (1 - e^(-lam dt)) (k s sigma - T)`.
7. **GNSS gate lock-out.** With an unmodelled accelerometer bias the chi^2 gate rejected every fix and the filter
   diverged (velocity 22 m/s). After 5 rejections in a row the fix is fused anyway (as PX4's EKF2 does).
8. **Simulator ordering in Tier 0.** Sensors were sampled after the motors had already responded to the new command;
   fixed to the lockstep order, which removed most of a 3 % thrust-gain bias.
9. **EKF2 GPS delay.** PX4 assumes 110 ms; Pegasus' GPS has none: `EKF2_GPS_DELAY = 0`.
10. **Two clocks in the simulator.** Pegasus only starts advancing PX4's clock (`time_usec`) once PX4 has connected,
    so it runs 1.28 s behind the simulator's own time. The force injector used simulator time while the offboard node
    flies on PX4 time: every force event came 1.28 s early relative to the trajectory and to the scenario's phase
    windows (the truth log was still right, so estimation errors were valid, but event metrics were not). The injector
    now runs on PX4's clock, and ingest checks every run: applied force vs schedule, max deviation 0.0 N.
11. **Recorder processes not stopped.** Background jobs of a non-interactive shell ignore SIGINT and `kill` hit the
    `bash -c` wrapper, so each `ros2 bag record` kept recording later flights on the same DDS domain. Jobs are now
    `exec`'d and stopped with SIGTERM; no process outlives its run.

## Things that are real effects (kept, and discussed)

1. **PX4's attitude estimate is tilted ~1.6 deg** in flight (EKF2 converges to a wrong tilt/accelerometer-bias pair
   during the climb; the pair is unobservable in hover). The force filter takes attitude as exact, so thrust (16 N) x
   tilt (0.028 rad) appears as a ~0.45 N lateral force. Magnetometer fusion mode and GPS delay were ruled out.
   Estimating a tilt error (EKF-C, 2 extra states) helps one seed and hurts another: with only vertical excitation it is
   weakly observable. Reported as a limitation; "attitude in the state" is future work.
2. **Thrust-gain / force coupling.** A force step and a thrust-gain change look the same unless the throttle varies;
   during the +5 N step EKF-B books part of it as k_b (F_z under-reads ~0.7 N in Tier 0 until the next excitation).
   sigma_k stays at 4-5 % of k: the filter reports this honestly (NEES consistent).
3. **Voltage noise biases k_b** (errors-in-variables: the filter treats V as an exact input). Voltage noise x5 -> k_b
   -9 %, F_z RMSE 0.43 -> 1.5 N. PX4's `battery_status` over DDS is rate-limited to 1 Hz: F_z RMSE 0.43 -> 0.75 N.
   The voltage must come at >= 50 Hz (ULog, or a raised DDS rate).
4. **Motor lag is on rotor speed, not thrust.** T ~ omega^2 makes a large thrust step faster than first order: the
   bench step test reads tau = 44 ms for a true 40 ms, and the filter's lambda settles ~5 % (Tier 0) to ~14 % (Tier 1)
   low. It does not bias the force.
5. **Bench prior pseudo-measurement hurts** when the bench value is off (+5 %): F_z bias 0.04 -> 0.7 N. Not used.
6. **Accelerometer bias state is required**: without it lateral RMSE 0.41 vs 0.12 N, force NEES 19 vs 2.1 (Tier 0).

## Decisions

- D1 (actuator_motors over DDS): not needed for the offline evaluation (ULog). Open for the optional live node.
- D2 (PX4 build): existing `px4_sitl_raptor` with `MC_RAPTOR_ENABLE = 0`; stock controllers fly.
- Priors: from the simulated bench test (`config/priors_bench.yaml`), perturbed k +5 %, tau +20 %.
- Tuning: Tier 0 on seed 0 of S1/S3/S4, then frozen. Reported numbers: seeds 1-4.
