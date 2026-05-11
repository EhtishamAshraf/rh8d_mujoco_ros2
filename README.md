# RH8D Hand Simulation (ROS 2 + MuJoCo)

This project simulates the **RH8D tendon-driven robotic hand** using **MuJoCo** and **ROS 2**. The simulation is controlled directly through a ROS 2 node that:

1. Loads the MuJoCo MJCF model.
2. Opens the MuJoCo viewer.
3. Subscribes to hand command messages.
4. Writes commands directly to MuJoCo actuators.
5. Publishes joint states, tendon states, fingertip forces, and contact status.

The main advantage of using MuJoCo is that it supports **native spatial tendon modeling**, which is more suitable for simulating the tendon-driven RH8D hand than Gazebo mimic-joint approximation.

Tested on:

- ROS 2 Humble
- MuJoCo

Demo video:
[![Demo Video](https://github.com/EhtishamAshraf/rh8d_mujoco_ros2/blob/42d79a12e8eb45b62df5007362b32549ec2f6697/assets/ros2_cylinder.png)](https://youtu.be/hQMjRyygGjY)

---

## Build and Launch Instructions

Go to the workspace. Build the packages:

```bash
colcon build
```

Source the workspace:

```bash
source install/setup.bash
```
Run the nodes:
```bash
ros2 run rh8d_mujoco_sim rh8d_mujoco_node --ros-args -p model_path:=/home/ehtisham/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/ros2/v2_rh8d_ws/src/rh8d_mujoco_sim/assets/mjcf/scene.xml

```

```bash
ros2 run rh8d_mujoco_sim rh8d_hand_test
```

---

## MuJoCo Model

The hand model is defined using an MJCF XML file:

```text
assets/mjcf/scene.xml
```

The model contains:

- RH8D hand body structure
- STL visual meshes
- Collision geoms
- Revolute joints
- Spatial tendons
- Tendon actuators
- Tendon position sensors
- Actuator force sensors
- Palm rangefinder sensor
- Fingertip force sensors
- Contact exclusions

In the MJCF file, the mesh directory is defined as below. The mesh folder must be located relative to the MJCF file.

```xml
<compiler angle="radian" meshdir="../meshes"/>
```

#### Tendon Modeling

The RH8D hand is tendon-driven. In MuJoCo, this is modeled using **spatial tendons**. This is more physically meaningful than the Gazebo version, where tendon behavior had to be approximated using mimic joints.

```xml
<tendon>
  <spatial name="idx_flex_spatial">
    <site site="palm_flexor_origin_idx"/>
    <site site="idx_guide_p"/>
    <site site="idx_guide_m"/>
    <site site="idx_guide_d"/>
    <site site="idx_anchor"/>
  </spatial>
</tendon>
```

The ring and small fingers are coupled using a MuJoCo equality constraint. This means the small finger tendon follows the ring finger tendon.

```xml
<equality>
  <tendon tendon1="sml_flex_spatial" tendon2="ring_flex_spatial"
          polycoef="0 1 0 0 0"
          solref="0.08 1.0"
          solimp="0.90 0.99 0.002"/>
</equality>
```

---

## ROS 2 Nodes

This project contains two main ROS 2 nodes.

---

#### 1. RH8D MuJoCo Simulation Node

Main responsibility:

- Load the MJCF model.
- Start the MuJoCo passive viewer.
- Subscribe to hand commands.
- Write commands directly to MuJoCo actuators.
- Step the MuJoCo simulation.
- Publish hand state.
- Publish fingertip forces.
- Publish contact status.

#### 2. RH8D Gesture / Object Pick Controller

Main responsibility:

- Publish gesture commands to `/rh8d/command`.
- Subscribe to `/rh8d/state`.
- Subscribe to `/rh8d/contacts`.
- Detect object presence using the palm range sensor.
- Close the hand only when an object is detected.
- Detect stable fingertip contacts.
- Latch tendon commands after grasp contact.
- Execute a predefined gesture sequence.

#### Flow of the controller

The controller does not immediately close the hand. Before closing, it checks whether an object is detected by the palm rangefinder. If the object is not detected, the hand stays open.

```python
range_threshold = 0.25
```

During the hand close gesture, the controller monitors fingertip contacts. Stable contact is required for multiple control cycles:

```python
required_contact_cycles = 5
```

The MuJoCo simulation node detects contact using fingertip force sensors. Contact is considered true when the Z-force magnitude is greater than:

```python
contact_threshold = 0.50
```

After stable grasp contact is detected, the controller latches the current tendon values. This prevents the hand from continuing to close after the object is already grasped.

---
