#!/usr/bin/env python3
"""
Code to print tendon lengths and forces to verify the hand’s configuration.
"""

import argparse
import mujoco

# Function for reading 3D vector sensor like 3-axis force sensor:
def sensor_scalar(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Sensor '{name}' not found")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    if dim != 1:
        raise ValueError(f"Sensor '{name}' is not scalar")
    return float(data.sensordata[adr])

# Function for getting the index of a named actuator in the MuJoCo model:
def actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator '{name}' not found")
    return aid

# Set pose for each finger
def set_pose(data, a_thumb_axis, a_index, a_middle, a_ring_small, a_thumb,
             thumb_axis, index, middle, ring_small, thumb):
    data.ctrl[a_thumb_axis] = thumb_axis
    data.ctrl[a_index] = index
    data.ctrl[a_middle] = middle
    data.ctrl[a_ring_small] = ring_small
    data.ctrl[a_thumb] = thumb

# Advance the simulation step by step
def settle(model, data, steps):
    for _ in range(steps):
        mujoco.mj_step(model, data)

# Print Tendons length and actuators force:
def print_state(model, data, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("TENDON LENGTHS")
    print(f"len_index  : {sensor_scalar(model, data, 'len_index'):.6f}")
    print(f"len_middle : {sensor_scalar(model, data, 'len_middle'):.6f}")
    print(f"len_ring   : {sensor_scalar(model, data, 'len_ring'):.6f}")
    print(f"len_small  : {sensor_scalar(model, data, 'len_small'):.6f}")
    print(f"len_thumb  : {sensor_scalar(model, data, 'len_thumb'):.6f}")

    print("\nACTUATOR FORCES")
    print(f"frc_index  : {sensor_scalar(model, data, 'frc_index'):.6f}")
    print(f"frc_middle : {sensor_scalar(model, data, 'frc_middle'):.6f}")
    print(f"frc_ring   : {sensor_scalar(model, data, 'frc_ring'):.6f}")
    print(f"frc_thumb  : {sensor_scalar(model, data, 'frc_thumb'):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Path to XML scene/model")
    ap.add_argument("--settle_steps", type=int, default=3000)

    # Open pose
    ap.add_argument("--open_thumb_axis", type=float, default=0.0)
    ap.add_argument("--open_index", type=float, default=0.0)
    ap.add_argument("--open_middle", type=float, default=0.0)
    ap.add_argument("--open_ring_small", type=float, default=0.0)
    ap.add_argument("--open_thumb", type=float, default=0.0)

    # Close pose
    ap.add_argument("--close_thumb_axis", type=float, default=1.57)
    ap.add_argument("--close_index", type=float, default=1.57)
    ap.add_argument("--close_middle", type=float, default=1.57)
    ap.add_argument("--close_ring_small", type=float, default=1.57)
    ap.add_argument("--close_thumb", type=float, default=1.57)

    args = ap.parse_args()

    # Loading MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    # Getting Actuator's index for all fingers
    a_thumb_axis = actuator_id(model, "pos_thumb_axis")
    a_index = actuator_id(model, "pos_index_tendon")
    a_middle = actuator_id(model, "pos_middle_tendon")
    a_ring_small = actuator_id(model, "pos_ring_small_tendon")
    a_thumb = actuator_id(model, "pos_thumb_tendon")

    # OPEN POSE
    set_pose(
        data, a_thumb_axis, a_index, a_middle, a_ring_small, a_thumb,
        args.open_thumb_axis, args.open_index, args.open_middle,
        args.open_ring_small, args.open_thumb
    )
    mujoco.mj_forward(model, data)
    settle(model, data, args.settle_steps)
    print_state(model, data, "OPEN POSE")

    # CLOSE POSE
    set_pose(
        data, a_thumb_axis, a_index, a_middle, a_ring_small, a_thumb,
        args.close_thumb_axis, args.close_index, args.close_middle,
        args.close_ring_small, args.close_thumb
    )
    mujoco.mj_forward(model, data)
    settle(model, data, args.settle_steps)
    print_state(model, data, "CLOSE POSE")


if __name__ == "__main__":
    main()