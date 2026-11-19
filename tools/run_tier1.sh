#!/usr/bin/env bash
# One Tier 1 flight, end to end: XRCE agent, rosbag2, offboard node, Isaac Sim + Pegasus + PX4 SITL, then ingest.
#   tools/run_tier1.sh S1_main 1            # scenario, seed
#   GUI=1 tools/run_tier1.sh S1_main 1      # with the Isaac window
#   T0=30 tools/run_tier1.sh ...            # scenario start [s of sim time] (default config/sim_tier1.yaml)
# Output: runs/tier1/<scenario>/seed<N>/ {truth_raw.*, meta*.yaml, px4/ (ULog, px4.log), bag/, *.log, *.parquet}
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCEN="${1:-S1_main}"; SEED="${2:-1}"
RUN="$ROOT/runs/tier1/$SCEN/seed$SEED"
T0="${T0:-$(python3 -c "import yaml;print(yaml.safe_load(open('$ROOT/config/sim_tier1.yaml'))['t_scenario_start'])")}"
DOMAIN=$(python3 -c "import yaml;print(yaml.safe_load(open('$ROOT/config/px4_params.yaml'))['ros_domain_id'])")
PORT=$(python3 -c "import yaml;print(yaml.safe_load(open('$ROOT/config/px4_params.yaml'))['uxrce_dds_port'])")
AGENT="$ROOT/third_party/xrce_agent/install"
rm -rf "$RUN"; mkdir -p "$RUN"
pids=()
cleanup() {
    # Every background job was started with exec, so these PIDs are the processes themselves (not bash wrappers).
    # SIGTERM, not SIGINT: background jobs of a non-interactive shell start with SIGINT ignored. ros2 bag closes
    # its file and writes metadata.yaml on SIGTERM.
    for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
    for p in "${pids[@]}"; do
        for _ in $(seq 1 50); do kill -0 "$p" 2>/dev/null || break; sleep 0.2; done
        kill -KILL "$p" 2>/dev/null || true
    done
    pids=()
    pkill -TERM -f "lib/efe_ros/[o]ffboard_node" 2>/dev/null || true
}
trap cleanup EXIT

ros() {  # a ROS 2 command on our domain, with our workspace; exec so the PID is the command itself
    exec env -u PYTHONPATH ROS_DOMAIN_ID="$DOMAIN" bash -c "source /opt/ros/jazzy/setup.bash && source '$ROOT/ros2_ws/install/setup.bash' && exec $*"
}

LD_LIBRARY_PATH="$AGENT/lib:$LD_LIBRARY_PATH" "$AGENT/bin/MicroXRCEAgent" udp4 -p "$PORT" > "$RUN/agent.log" 2>&1 &
pids+=($!)
TOPICS="/fmu/out/sensor_combined /fmu/out/vehicle_attitude /fmu/out/vehicle_odometry /fmu/out/vehicle_gps_position \
/fmu/out/vehicle_local_position_v1 /fmu/out/vehicle_status_v4 /fmu/out/battery_status_v1 /fmu/out/actuator_motors"
(ros "ros2 bag record -o '$RUN/bag' $TOPICS") > "$RUN/bag.log" 2>&1 &
pids+=($!)
(ros "ros2 run efe_ros offboard_node --ros-args -p scenario:=$SCEN -p t0:=$T0 -p repo:=$ROOT") > "$RUN/offboard.log" 2>&1 &
pids+=($!)

HEADLESS="--headless"; [ -n "$GUI" ] && HEADLESS=""
echo "==> $SCEN seed $SEED: Isaac + PX4 (scenario at t = $T0 s), logs in $RUN"
env -u PYTHONPATH "$HOME/isaacsim/python.sh" "$ROOT/sim/efe_sim.py" --scenario "$SCEN" --seed "$SEED" --t0 "$T0" \
    --out "$RUN" $HEADLESS 2>&1 | grep --line-buffered -E "^\[efe_sim\]|Error|error|Traceback" | tee "$RUN/sim.log"
cleanup; trap - EXIT
echo "==> offboard: $(grep -E 'armed|scenario|landed|warn' "$RUN/offboard.log" | tr '\n' ' ' | cut -c1-300)"
(ros "python3 '$ROOT/tools/bag_to_csv.py' '$RUN'") > "$RUN/bag_csv.log" 2>&1 || echo "bag_to_csv failed, see $RUN/bag_csv.log"
env -u PYTHONPATH "$ROOT/.venv/bin/python" -m efe.io.ingest "$RUN"
env -u PYTHONPATH "$ROOT/.venv/bin/python" -m efe.io.ingest --check-bag "$RUN" || true
