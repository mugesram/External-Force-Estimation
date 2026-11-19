#!/usr/bin/env bash
# One-time setup of the ROS 2 side, all inside this repository:
#   ros2_ws/src/px4_msgs   PX4's px4_msgs package, with the message definitions of OUR PX4 tree (they must match)
#   third_party/xrce_agent Micro XRCE-DDS Agent v2.4.3 (the bridge PX4's uxrce_dds_client talks to)
#   ros2_ws                colcon build of px4_msgs + efe_ros
# Usage: tools/setup_ros.sh [PX4_DIR]   (default ~/Documents/Repos/PX4-Autopilot)
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${1:-$HOME/Documents/Repos/PX4-Autopilot}"
source /opt/ros/jazzy/setup.bash

MSGS="$ROOT/ros2_ws/src/px4_msgs"
if [ ! -d "$MSGS" ]; then
    git clone --depth 1 https://github.com/PX4/px4_msgs.git "$MSGS"
fi
rm -f "$MSGS"/msg/*.msg "$MSGS"/msg/versioned/*.msg "$MSGS"/srv/*.srv
cp "$PX4_DIR"/msg/*.msg "$MSGS"/msg/
cp "$PX4_DIR"/msg/versioned/*.msg "$MSGS"/msg/
cp "$PX4_DIR"/srv/*.srv "$MSGS"/srv/
echo "px4_msgs: $(ls "$MSGS"/msg | wc -l) messages from $PX4_DIR ($(git -C "$PX4_DIR" describe --tags))"

AGENT="$ROOT/third_party/xrce_agent"
if [ ! -x "$AGENT/install/bin/MicroXRCEAgent" ]; then
    mkdir -p "$ROOT/third_party"
    [ -d "$AGENT/src" ] || git clone --depth 1 --branch v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$AGENT/src"
    cmake -S "$AGENT/src" -B "$AGENT/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$AGENT/install" \
          -DUAGENT_BUILD_EXECUTABLE=ON -DUAGENT_USE_SYSTEM_FASTDDS=OFF -DUAGENT_USE_SYSTEM_FASTCDR=OFF
    cmake --build "$AGENT/build" -j"$(nproc)"
    cmake --install "$AGENT/build"
fi
echo "agent: $AGENT/install/bin/MicroXRCEAgent"

cd "$ROOT/ros2_ws"
env -u PYTHONPATH bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"
echo "done: source ros2_ws/install/setup.bash"
