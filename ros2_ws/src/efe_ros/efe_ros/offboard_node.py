"""Flies a scenario through PX4 offboard mode over uXRCE-DDS.

    ros2 run efe_ros offboard_node --ros-args -p scenario:=S1_main -p t0:=30.0 -p repo:=/path/to/External-Force-Estimation

Timeline, on PX4's clock (in lockstep that is the simulator's clock, the same one the force injector uses):
  wait for a valid local position -> stream setpoints -> offboard + arm -> climb to the scenario's first point ->
  hold until t0 -> follow the scenario trajectory (position, velocity and acceleration feed-forward) for its duration ->
  land -> exit.
The trajectory comes from config/scenarios/<scenario>.yaml via efe.scenario, the same code the Tier 0 sim uses.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import rclpy
from px4_msgs.msg import OffboardControlMode, SensorCombined, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def topic(name, msg_cls):
    """PX4 names the topic of a versioned message <name>_v<N> when N > 0."""
    v = int(getattr(msg_cls, "MESSAGE_VERSION", 0))
    return f"{name}_v{v}" if v > 0 else name


class OffboardNode(Node):
    def __init__(self):
        super().__init__("efe_offboard")
        self.declare_parameter("scenario", "S1_main")
        self.declare_parameter("t0", 30.0)
        self.declare_parameter("repo", str(Path(__file__).resolve().parents[4]))
        self.declare_parameter("rate_hz", 50.0)
        repo = self.get_parameter("repo").value
        sys.path.insert(0, repo)
        from efe.scenario import Trajectory, load_scenario

        self.sc = load_scenario(self.get_parameter("scenario").value)
        self.traj = Trajectory(self.sc["trajectory"])
        self.t0 = float(self.get_parameter("t0").value)
        self.duration = float(self.sc["duration"])

        self.pub_mode = self.create_publisher(OffboardControlMode, topic("/fmu/in/offboard_control_mode", OffboardControlMode), PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, topic("/fmu/in/trajectory_setpoint", TrajectorySetpoint), PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, topic("/fmu/in/vehicle_command", VehicleCommand), PX4_QOS)
        self.create_subscription(VehicleStatus, topic("/fmu/out/vehicle_status", VehicleStatus), self._status, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, topic("/fmu/out/vehicle_local_position", VehicleLocalPosition),
                                 self._lpos, PX4_QOS)
        self.create_subscription(SensorCombined, "/fmu/out/sensor_combined", self._imu, PX4_QOS)

        self.status = None
        self.lpos = None
        self.px4_us = 0
        self.state = "WAIT"
        self.counter = 0
        self.yaw = float("nan")
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(f"scenario {self.sc['name']}: trajectory from t0 = {self.t0:.1f} s (PX4 time) for "
                               f"{self.duration:.0f} s")

    # --- callbacks ------------------------------------------------------------------------------------------
    def _status(self, msg):
        self.status = msg

    def _lpos(self, msg):
        self.lpos = msg

    def _imu(self, msg):
        self.px4_us = int(msg.timestamp)

    @property
    def t_px4(self):
        return self.px4_us * 1e-6

    # --- commands -------------------------------------------------------------------------------------------
    def _command(self, command, **params):
        m = VehicleCommand()
        m.command = command
        for i in range(1, 8):
            setattr(m, f"param{i}", float(params.get(f"param{i}", 0.0)))
        m.target_system, m.target_component = 1, 1
        m.source_system, m.source_component = 1, 1
        m.from_external = True
        m.timestamp = self.px4_us
        self.pub_cmd.publish(m)

    def _stream(self, p, v=None, a=None):
        mode = OffboardControlMode()
        mode.timestamp = self.px4_us
        mode.position = True
        mode.velocity = v is not None
        mode.acceleration = a is not None
        self.pub_mode.publish(mode)
        sp = TrajectorySetpoint()
        sp.timestamp = self.px4_us
        nan3 = [float("nan")] * 3
        sp.position = [float(x) for x in p]
        sp.velocity = [float(x) for x in v] if v is not None else nan3
        sp.acceleration = [float(x) for x in a] if a is not None else nan3
        sp.yaw = self.yaw
        self.pub_sp.publish(sp)

    # --- state machine --------------------------------------------------------------------------------------
    def _tick(self):
        if self.px4_us == 0 or self.lpos is None or self.status is None:
            return
        p0, _, _ = self.traj(0.0)
        if self.state == "WAIT":
            if self.lpos.xy_valid and self.lpos.z_valid:
                self.yaw = float(self.lpos.heading) if math.isfinite(self.lpos.heading) else 0.0
                self.ground = [self.lpos.x, self.lpos.y, self.lpos.z]
                self.state = "PREARM"
                self.get_logger().info(f"local position valid at t = {self.t_px4:.1f} s, heading {math.degrees(self.yaw):.0f} deg")
            return
        if self.state == "PREARM":
            self._stream(self.ground)
            self.counter += 1
            if self.counter == 20:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            if self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED and \
                    self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.state = "CLIMB"
                self.get_logger().info(f"armed in offboard at t = {self.t_px4:.1f} s, climbing to {p0}")
            elif self.counter % 100 == 0:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            return
        if self.state == "CLIMB":
            self._stream(p0)
            if self.t_px4 >= self.t0:
                err = math.dist([self.lpos.x, self.lpos.y, self.lpos.z], p0)
                if err > 0.3:
                    self.get_logger().warn(f"scenario starts with the vehicle {err:.2f} m from its first point")
                self.state = "SCENARIO"
                self.get_logger().info(f"scenario start at t = {self.t_px4:.2f} s")
            return
        if self.state == "SCENARIO":
            ts = self.t_px4 - self.t0
            p, v, a = self.traj(ts)
            self._stream(p, v, a)
            if ts >= self.duration:
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.state = "LAND"
                self.get_logger().info(f"scenario done at t = {self.t_px4:.1f} s, landing")
            return
        if self.state == "LAND":
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED or self.t_px4 > self.t0 + self.duration + 20:
                self.get_logger().info("landed")
                raise SystemExit(0)


def main():
    rclpy.init()
    node = OffboardNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
