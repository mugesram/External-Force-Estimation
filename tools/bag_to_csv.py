#!/usr/bin/env python3
"""Convert the DDS recording (rosbag2) of a run to CSV, with ROS 2's own Python (rosbag2_py, px4_msgs).

    source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
    python3 tools/bag_to_csv.py runs/tier1/S1_main/seed1          # -> runs/.../bag_csv/<topic>.csv

Only numeric fields (scalars and fixed arrays) are written. efe.io.ingest --check-bag then compares these streams
with the ULog sample by sample (plan gate P4: the DDS path carries the same data as the log).
"""

import csv
import subprocess
import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def flatten(msg):
    out = {}
    for name in msg.get_fields_and_field_types():
        v = getattr(msg, name)
        if isinstance(v, (int, float, bool)):
            out[name] = float(v)
        elif hasattr(v, "__len__") and not isinstance(v, str) and len(v) <= 16:
            try:
                for i, x in enumerate(v):
                    out[f"{name}[{i}]"] = float(x)
            except (TypeError, ValueError):
                pass
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "bag" / "metadata.yaml").exists():
        # the recorder was stopped without writing its index: rebuild it from the MCAP file
        subprocess.run(["ros2", "bag", "reindex", str(run_dir / "bag"), "-s", "mcap"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(run_dir / "bag"), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    rows = {t: [] for t in types}
    while reader.has_next():
        topic, data, t_recv = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        r = flatten(msg)
        r["recv_ns"] = float(t_recv)
        rows[topic].append(r)
    out = run_dir / "bag_csv"
    out.mkdir(exist_ok=True)
    for topic, rs in rows.items():
        if not rs:
            continue
        name = topic.strip("/").replace("/", "__")
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
        print(f"{topic}: {len(rs)} messages")


if __name__ == "__main__":
    main(sys.argv[1])
