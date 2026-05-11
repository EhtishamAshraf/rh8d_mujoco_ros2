#!/usr/bin/env python3

"""
Note:
    Thumb normal magnitude is big because it balances the four fingers together.
    Thumb shear is slightly larger than normal because the contact is oblique, not a perfect flat-face normal contact

Run the file with: 
    python3 pick.py --scene /home/ehtisham/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/ros2/v2_rh8d_ws/src/rh8d_mujoco_sim/assets/mjcf/scene.xml

"""

import argparse
import math
import numpy as np
import mujoco
import mujoco.viewer

# Function for getting the index of a named actuator in the MuJoCo model:
def actuator_id(m, name):
    x = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if x < 0:
        raise ValueError(f"Actuator '{name}' not found")
    return x

# Function for getting the index of a named joint in the MuJoCo model:
def joint_id(m, name):
    x = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if x < 0:
        raise ValueError(f"Joint '{name}' not found")
    return x

# Function for getting the position of a specific joint:
def joint_qpos(m, d, name):
    j = joint_id(m, name)
    return float(d.qpos[m.jnt_qposadr[j]])

# Function for reading single value sensor like range sensor:
def sensor_scalar(m, d, name):
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if s < 0:
        raise ValueError(f"Sensor '{name}' not found")
    adr = m.sensor_adr[s]
    dim = m.sensor_dim[s]
    if dim != 1:
        raise ValueError(f"Sensor '{name}' is not scalar")
    return float(d.sensordata[adr])

# Function for reading 3D vector sensor like 3-axis force sensor:
def sensor_vec3(m, d, name):
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if s < 0:
        raise ValueError(f"Sensor '{name}' not found")
    adr = m.sensor_adr[s]
    dim = m.sensor_dim[s]
    if dim < 3:
        raise ValueError(f"Sensor '{name}' is not 3D")
    return np.array(d.sensordata[adr:adr + 3], dtype=float)

# Function for computing grasp metrics from a fingertip force vector.
def metrics(v):
    fx, fy, fz = abs(float(v[0])), abs(float(v[1])), abs(float(v[2]))
    shear = math.sqrt(fx * fx + fy * fy)
    normal = abs(fz)
    slip_ratio = shear / (normal + 1e-6)
    return fx, fy, fz, shear, normal, slip_ratio

def header(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

def print_row(label, vec, count, need):
    fx, fy, fz, shear, normal, slip = metrics(vec)
    contact = count >= need
    print(
        f"{label:<8s}"
        f"{fx:>11.6f}"
        f"{fy:>11.6f}"
        f"{fz:>11.6f}"
        f"{('YES' if contact else 'NO'):>11s}"
        f"{count:>8d}"
    )

def main():
    
    # Using argparse, which is a standard Python module to handle command-line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)

    # Open/close commands
    ap.add_argument("--open_fingers", type=float, default=0.0)
    ap.add_argument("--open_thumb", type=float, default=0.0)
    ap.add_argument("--close_fingers", type=float, default=1.57)
    ap.add_argument("--close_thumb", type=float, default=1.57)

    # Thumb axis first
    ap.add_argument("--thumb_axis_target", type=float, default=1.57)
    ap.add_argument("--thumb_axis_rate", type=float, default=1.2)
    ap.add_argument("--thumb_axis_tol", type=float, default=0.02)

    # Object detection
    ap.add_argument("--range_threshold", type=float, default=0.25)

    # RAW XML contact thresholds
    ap.add_argument("--normal_contact_threshold", type=float, default=0.50)
    ap.add_argument("--thumb_contact_threshold", type=float, default=0.50)
    ap.add_argument("--contact_hold_frames", type=int, default=8)

    # Closing speeds
    ap.add_argument("--rate_index", type=float, default=0.5)
    ap.add_argument("--rate_middle", type=float, default=0.5)
    ap.add_argument("--rate_thumb", type=float, default=0.5)
    ap.add_argument("--rate_ring_small", type=float, default=0.5)

    ap.add_argument("--print_hz", type=float, default=5.0)
    args = ap.parse_args()

    # Loading MuJoCo model and data
    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    print_dt = 1.0 / args.print_hz

    # Getting Actuator's index for all fingers
    a_thumb_axis = actuator_id(m, "pos_thumb_axis")
    a_index = actuator_id(m, "pos_index_tendon")
    a_middle = actuator_id(m, "pos_middle_tendon")
    a_ring_small = actuator_id(m, "pos_ring_small_tendon")
    a_thumb = actuator_id(m, "pos_thumb_tendon")

    # Start fully open
    d.ctrl[a_thumb_axis] = 0.0
    d.ctrl[a_index] = args.open_fingers
    d.ctrl[a_middle] = args.open_fingers
    d.ctrl[a_ring_small] = args.open_fingers
    d.ctrl[a_thumb] = args.open_thumb
    mujoco.mj_forward(m, d)

    cnt_idx = 0
    cnt_mid = 0
    cnt_ring = 0
    cnt_small = 0
    cnt_th = 0

    phase = "MOVE_THUMB_AXIS"
    last_print = -1.0

    header("RAW XML FORCE BASED PICK")
    print(f"Contact rule (index/middle/ring/small): normal = abs(Fz) >= {args.normal_contact_threshold}")
    print(f"Contact rule (thumb)                  : normal = abs(Fz) >= {args.thumb_contact_threshold}")
    print(f"Sustained contact frames              : {args.contact_hold_frames}")
    print("Ring+Small shared rule                : if either ring or small contacts, shared actuator stops")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            t = float(d.time)

            # checking if object is present or not
            rf = sensor_scalar(m, d, "palm_range")
            object_present = (rf > 0.0) and (rf < args.range_threshold)

            # RAW XML force sensors
            v_idx = sensor_vec3(m, d, "index_tip_force")
            v_mid = sensor_vec3(m, d, "middle_tip_force")
            v_ring = sensor_vec3(m, d, "ring_tip_force")
            v_small = sensor_vec3(m, d, "small_tip_force")
            v_th = sensor_vec3(m, d, "thumb_tip_force")

            # Metrics
            _, _, _, _, n_idx, _ = metrics(v_idx)
            _, _, _, _, n_mid, _ = metrics(v_mid)
            _, _, _, _, n_ring, _ = metrics(v_ring)
            _, _, _, _, n_small, _ = metrics(v_small)
            _, _, _, _, n_th, _ = metrics(v_th)

            # Update contact counters
            cnt_idx = cnt_idx + 1 if n_idx >= args.normal_contact_threshold else 0
            cnt_mid = cnt_mid + 1 if n_mid >= args.normal_contact_threshold else 0
            cnt_ring = cnt_ring + 1 if n_ring >= args.normal_contact_threshold else 0
            cnt_small = cnt_small + 1 if n_small >= args.normal_contact_threshold else 0
            cnt_th = cnt_th + 1 if n_th >= args.thumb_contact_threshold else 0

            idx_contact = cnt_idx >= args.contact_hold_frames
            mid_contact = cnt_mid >= args.contact_hold_frames
            ring_contact = cnt_ring >= args.contact_hold_frames
            small_contact = cnt_small >= args.contact_hold_frames
            th_contact = cnt_th >= args.contact_hold_frames

            # Simple conditioning to detect and grab the object
            if phase == "MOVE_THUMB_AXIS":
                # Keep all tendons open while thumb axis moves
                d.ctrl[a_index] = args.open_fingers
                d.ctrl[a_middle] = args.open_fingers
                d.ctrl[a_ring_small] = args.open_fingers
                d.ctrl[a_thumb] = args.open_thumb

                d.ctrl[a_thumb_axis] = min(
                    float(d.ctrl[a_thumb_axis]) + args.thumb_axis_rate * dt,
                    args.thumb_axis_target
                )

                if abs(joint_qpos(m, d, "Thumb_axis--palmL:1") - args.thumb_axis_target) <= args.thumb_axis_tol:
                    phase = "CHECK_OBJECT"

            elif phase == "CHECK_OBJECT":
                if object_present:
                    # Reset counters so no false contact carries into grasp phase
                    cnt_idx = 0
                    cnt_mid = 0
                    cnt_ring = 0
                    cnt_small = 0
                    cnt_th = 0
                    phase = "CLOSE_TO_PICK"
                else:
                    phase = "WAIT_NO_OBJECT"

            elif phase == "WAIT_NO_OBJECT":
                d.ctrl[a_index] = args.open_fingers
                d.ctrl[a_middle] = args.open_fingers
                d.ctrl[a_ring_small] = args.open_fingers
                d.ctrl[a_thumb] = args.open_thumb

                if object_present:
                    cnt_idx = 0
                    cnt_mid = 0
                    cnt_ring = 0
                    cnt_small = 0
                    cnt_th = 0
                    phase = "CLOSE_TO_PICK"

            elif phase == "CLOSE_TO_PICK":
                # Incremently increase the actuator command to close the fingers:
                # Index
                if not idx_contact:
                    d.ctrl[a_index] = min(float(d.ctrl[a_index]) + args.rate_index * dt, args.close_fingers)

                # # Middle
                if not mid_contact:
                    d.ctrl[a_middle] = min(float(d.ctrl[a_middle]) + args.rate_middle * dt, args.close_fingers)

                # Thumb tendon
                if not th_contact:
                    d.ctrl[a_thumb] = min(float(d.ctrl[a_thumb]) + args.rate_thumb * dt, args.close_thumb)

                # Ring+Small shared
                if not (ring_contact or small_contact):
                    d.ctrl[a_ring_small] = min(
                        float(d.ctrl[a_ring_small]) + args.rate_ring_small * dt,
                        args.close_fingers
                    )

                if th_contact and idx_contact and mid_contact and (ring_contact or small_contact):  # and (mid_contact or ring_contact or small_contact):
                    phase = "PICK_DONE"

            elif phase == "PICK_DONE":
                pass

            mujoco.mj_step(m, d)
            viewer.sync()

            if t - last_print >= print_dt:
                last_print = t
                header(f"STATUS | t = {t:8.3f} s | PHASE = {phase}")
                print(f"Range finder         : {rf: .6f}")
                print(f"Object present       : {'YES' if object_present else 'NO'}")
                print(f"Thumb axis joint     : {joint_qpos(m, d, 'Thumb_axis--palmL:1'): .6f}")
                print(f"pos_thumb_axis      : {float(d.ctrl[a_thumb_axis]): .6f}")

                print("\n[ACTUATOR CTRLS]")
                print(f"pos_index_tendon      : {float(d.ctrl[a_index]): .6f}")
                print(f"pos_middle_tendon     : {float(d.ctrl[a_middle]): .6f}")
                print(f"pos_ring_small_tendon : {float(d.ctrl[a_ring_small]): .6f}")
                print(f"pos_thumb_tendon      : {float(d.ctrl[a_thumb]): .6f}")

                print("\n[JOINT POSITIONS]")
                print(f"Index MCP             : {joint_qpos(m, d, 'Index_Proximal--palmL:1'): .6f}")
                print(f"Index PIP             : {joint_qpos(m, d, 'Index_Middle--Index_Proximal'): .6f}")
                print(f"Index DIP             : {joint_qpos(m, d, 'Index_Distal--Index_Middle'): .6f}")
                print(f"Middle MCP            : {joint_qpos(m, d, 'Middle_Proximal--palmL:1'): .6f}")
                print(f"Middle PIP            : {joint_qpos(m, d, 'Middle_Middle--Middle_Proximal'): .6f}")
                print(f"Middle DIP            : {joint_qpos(m, d, 'Middle_Distal--Middle_Middle'): .6f}")
                print(f"Ring MCP              : {joint_qpos(m, d, 'Ring_Proximal--palmL:1'): .6f}")
                print(f"Ring PIP              : {joint_qpos(m, d, 'Ring_Middle--Ring_Proximal'): .6f}")
                print(f"Ring DIP              : {joint_qpos(m, d, 'Ring_Distal--Ring_Middle'): .6f}")
                print(f"Small MCP             : {joint_qpos(m, d, 'Small_Proximal--palmL:1'): .6f}")
                print(f"Small PIP             : {joint_qpos(m, d, 'Small_Middle--Small_Proximal'): .6f}")
                print(f"Small DIP             : {joint_qpos(m, d, 'Small_Distal--Small_Middle'): .6f}")
                print(f"Thumb MCP             : {joint_qpos(m, d, 'Thumb_Methacarpal--Thumb_axis'): .6f}")
                print(f"Thumb PIP             : {joint_qpos(m, d, 'Thumb_Proximal--Thumb_Methacarpal'): .6f}")
                print(f"Thumb DIP             : {joint_qpos(m, d, 'Thumb_Distal--Thumb_Proximal'): .6f}")

                print("\n[RAW XML FORCE SUMMARY]")
                print(f"{'Finger':<8s}{'Fx':>11s}{'Fy':>11s}{'Fz':>11s}{'Contact':>11s}{'Count':>8s}")
                print("-" * 102)
                print_row("INDEX", v_idx, cnt_idx, args.contact_hold_frames)
                print_row("MIDDLE", v_mid, cnt_mid, args.contact_hold_frames)
                print_row("RING", v_ring, cnt_ring, args.contact_hold_frames)
                print_row("SMALL", v_small, cnt_small, args.contact_hold_frames)
                print_row("THUMB", v_th, cnt_th, args.contact_hold_frames)

if __name__ == "__main__":
    main()