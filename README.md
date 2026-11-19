# External Force Estimation with an EKF, without RPM

Mini project for *Implementing State Estimation Filters*: an Extended Kalman Filter that estimates the 3D external
force on a quadrotor from **motor commands (PWM), battery voltage, IMU and RTK GNSS**, with no rotor-speed sensing.
Thrust, thrust gain and motor lag are states, so battery sag and motor dynamics don't turn into a fake force.

- Design: `resources/plan.pdf`. Execution plan: `docs/EXECUTION_PLAN.md`. Measured facts and decisions: `docs/FINDINGS.md`.

## Layout

```
efe/estimation   frames, models + Jacobians, the EKF (KF / EKF-A / EKF-B / EKF-B+bias by config), momentum observer,
                 filter inputs (attitude, voltage low-pass), runner
efe/truth        truth models shared by both simulators: motors/ESC, battery, external-force schedule
efe/pysim        Tier 0: pure-Python flight simulator
efe/io           RunData (one flight's streams), Tier 1 ingest (ULog + truth -> RunData)
efe/evaluation   metrics (RMSE, bias, rise time, NEES/NIS), plots, run matrix, experiments, replay Monte Carlo
efe/tools        bench identification
sim/             Tier 1: Isaac Sim + Pegasus + PX4 launcher (efe_sim.py) and Pegasus subclasses (pegasus_ext/)
ros2_ws/         px4_msgs (from our PX4 tree) and efe_ros (offboard flight node)
config/          vehicle, battery, sensors, scenarios, filters, priors, PX4 parameters
tests/           pytest
```

## Quick start

```bash
make venv && make test          # Python side
make tier0                      # Tier 0: 5 scenarios x 5 filters x seeds 1-4 -> results/tier0/
make experiments                # robustness, ablations, 50-run replay Monte Carlo
make bench                      # bench identification of k_b, u0, tau, drag -> config/priors_bench.yaml

make ros-setup                  # once: px4_msgs, Micro XRCE-DDS Agent, colcon build
make tier1-run SCEN=S1_main SEED=1   # Isaac + Pegasus + PX4 SITL over DDS, then ingest
make tier1-eval SCENARIOS=S1_main SEEDS=1
```

Seed 0 is the tuning seed; reported numbers use seeds 1-4.

## External tools used (not modified)

Isaac Sim 5.1 (`~/isaacsim`), Pegasus Simulator (`~/Documents/Repos/PegasusSimulator`), PX4-Autopilot v1.18
(`~/Documents/Repos/PX4-Autopilot`, `px4_sitl_raptor` build with RAPTOR disabled), ROS 2 Jazzy.
