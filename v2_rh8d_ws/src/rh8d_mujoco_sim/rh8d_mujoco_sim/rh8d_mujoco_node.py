#!/usr/bin/env python3


"""
ROS2 node for simulating the RH8D robotic hand in MuJoCo.

Run with this command:
ros2 run rh8d_mujoco_sim rh8d_mujoco_node --ros-args -p model_path:=/home/ehtisham/Desktop/Robotics_uclv/03_PROJECTS/P1_rh8d_sim/2-MuJoCo/ros2/v2_rh8d_ws/src/rh8d_mujoco_sim/assets/mjcf/scene.xml
"""

import threading
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from rh8d_mujoco_interfaces.msg import HandCommand, HandState, FingertipForces, HandContacts


class RH8DMujocoNode(Node):

    def __init__(self):
        super().__init__("rh8d_mujoco_node")
        self.get_logger().info("Initialized Node: rh8d_mujoco_node")

        # Get the MuJoCo model XML file path from ROS2 parameter
        self.declare_parameter("model_path", "")
        self.model_path = self.get_parameter("model_path").value
        if not self.model_path:
            raise ValueError("Parameter 'model_path' must be set")

        # Allow callbacks to run concurrently (multi-threading)
        self.callback_group = ReentrantCallbackGroup()

        # Default ROS2 QoS settings for publishers/subscribers
        self.qos_default = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscriber for listening to hand commands
        self.command_sub = self.create_subscription(
            HandCommand,
            "/rh8d/command",
            self.command_callback,
            self.qos_default,
            callback_group=self.callback_group
        )

        # Publisher fpr hand state, fingertip forces, and contacts
        self.state_pub = self.create_publisher(
            HandState,
            "/rh8d/state",
            self.qos_default,
            callback_group=self.callback_group
        )
        self.force_pub = self.create_publisher(
            FingertipForces,
            "/rh8d/fingertip_forces",
            self.qos_default,
            callback_group=self.callback_group
        )
        self.contact_pub = self.create_publisher(
            HandContacts,
            "/rh8d/contacts",
            self.qos_default,
            callback_group=self.callback_group
        )
        
        # Load MuJoCo model and create data object
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.lock = threading.Lock()
        self.sim_active = True

        self.command = HandCommand()

        # Get actuators index
        self.a_forearm = self._actuator_id("pos_forearm")
        self.a_palm_axis = self._actuator_id("pos_palm_axis")
        self.a_palm_l = self._actuator_id("pos_palmL")
        self.a_thumb_axis = self._actuator_id("pos_thumb_axis")
        self.a_index = self._actuator_id("pos_index_tendon")
        self.a_middle = self._actuator_id("pos_middle_tendon")
        self.a_ring_small = self._actuator_id("pos_ring_small_tendon")
        self.a_thumb = self._actuator_id("pos_thumb_tendon")


        # Threshold for detecting fingertip contact
        self.contact_threshold = 0.50

        # Start simulation thread (runs independently)
        self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.sim_thread.start()

        self.get_logger().info("RH8D MuJoCo simulation started")

    # Function for getting the index of a named actuator in the MuJoCo model:
    def _actuator_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise ValueError(f"Actuator '{name}' not found")
        return idx

    # Function for getting the position of a specific joint:
    def _joint_qpos(self, name: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found")
        return float(self.data.qpos[self.model.jnt_qposadr[jid]])

    # Function for reading single value sensor like range sensor:
    def _sensor_scalar(self, name: str) -> float:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"Sensor '{name}' not found")
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        if dim != 1:
            raise ValueError(f"Sensor '{name}' is not scalar")
        return float(self.data.sensordata[adr])

    # Function for reading 3D vector sensor like 3-axis force sensor:
    def _sensor_vec3(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"Sensor '{name}' not found")
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        if dim < 3:
            raise ValueError(f"Sensor '{name}' is not 3D")
        return np.array(self.data.sensordata[adr:adr + 3], dtype=float)

    def command_callback(self, msg: HandCommand):
        with self.lock:
            self.command = msg

    # Function for writing commands to the actuators so the joints move in the MuJoCo simulation
    def write_actuators(self):
        with self.lock:
            cmd = self.command
            self.data.ctrl[self.a_forearm] = cmd.forearm
            self.data.ctrl[self.a_palm_axis] = cmd.palm_axis
            self.data.ctrl[self.a_palm_l] = cmd.palm_l
            self.data.ctrl[self.a_thumb_axis] = cmd.thumb_axis
            self.data.ctrl[self.a_index] = cmd.index_tendon
            self.data.ctrl[self.a_middle] = cmd.middle_tendon
            self.data.ctrl[self.a_ring_small] = cmd.ring_small_tendon
            self.data.ctrl[self.a_thumb] = cmd.thumb_tendon

    # Publish HandState, FingertipForces, and HandContacts messages.
    # Reads joint positions and sensor data from MuJoCo.
    def publish_state(self):
        state = HandState()

        state.forearm_joint = self._joint_qpos("forearm:1--base:1")
        state.palm_axis_joint = self._joint_qpos("palm_axis:1--forearm:1")
        state.palm_l_joint = self._joint_qpos("palmL:1--palm_axis:1")
        state.thumb_axis_joint = self._joint_qpos("Thumb_axis--palmL:1")

        state.thumb_axis_joint = self._joint_qpos("Thumb_axis--palmL:1")

        state.index_mcp = self._joint_qpos("Index_Proximal--palmL:1")
        state.index_pip = self._joint_qpos("Index_Middle--Index_Proximal")
        state.index_dip = self._joint_qpos("Index_Distal--Index_Middle")

        state.middle_mcp = self._joint_qpos("Middle_Proximal--palmL:1")
        state.middle_pip = self._joint_qpos("Middle_Middle--Middle_Proximal")
        state.middle_dip = self._joint_qpos("Middle_Distal--Middle_Middle")

        state.ring_mcp = self._joint_qpos("Ring_Proximal--palmL:1")
        state.ring_pip = self._joint_qpos("Ring_Middle--Ring_Proximal")
        state.ring_dip = self._joint_qpos("Ring_Distal--Ring_Middle")

        state.small_mcp = self._joint_qpos("Small_Proximal--palmL:1")
        state.small_pip = self._joint_qpos("Small_Middle--Small_Proximal")
        state.small_dip = self._joint_qpos("Small_Distal--Small_Middle")

        state.thumb_mcp = self._joint_qpos("Thumb_Methacarpal--Thumb_axis")
        state.thumb_pip = self._joint_qpos("Thumb_Proximal--Thumb_Methacarpal")
        state.thumb_dip = self._joint_qpos("Thumb_Distal--Thumb_Proximal")

        state.len_index = self._sensor_scalar("len_index")
        state.len_middle = self._sensor_scalar("len_middle")
        state.len_ring = self._sensor_scalar("len_ring")
        state.len_small = self._sensor_scalar("len_small")
        state.len_thumb = self._sensor_scalar("len_thumb")

        state.frc_index = self._sensor_scalar("frc_index")
        state.frc_middle = self._sensor_scalar("frc_middle")
        state.frc_ring = self._sensor_scalar("frc_ring")
        state.frc_thumb = self._sensor_scalar("frc_thumb")

        state.palm_range = self._sensor_scalar("palm_range")

        self.state_pub.publish(state)

        forces = FingertipForces()
        v = self._sensor_vec3("index_tip_force")
        forces.index.x, forces.index.y, forces.index.z = float(v[0]), float(v[1]), float(v[2])

        v = self._sensor_vec3("middle_tip_force")
        forces.middle.x, forces.middle.y, forces.middle.z = float(v[0]), float(v[1]), float(v[2])

        v = self._sensor_vec3("ring_tip_force")
        forces.ring.x, forces.ring.y, forces.ring.z = float(v[0]), float(v[1]), float(v[2])

        v = self._sensor_vec3("small_tip_force")
        forces.small.x, forces.small.y, forces.small.z = float(v[0]), float(v[1]), float(v[2])

        v = self._sensor_vec3("thumb_tip_force")
        forces.thumb.x, forces.thumb.y, forces.thumb.z = float(v[0]), float(v[1]), float(v[2])

        self.force_pub.publish(forces)

        contacts = HandContacts()
        contacts.index = bool(abs(float(self._sensor_vec3("index_tip_force")[2])) >= self.contact_threshold)
        contacts.middle = bool(abs(float(self._sensor_vec3("middle_tip_force")[2])) >= self.contact_threshold)
        contacts.ring = bool(abs(float(self._sensor_vec3("ring_tip_force")[2])) >= self.contact_threshold)
        contacts.small = bool(abs(float(self._sensor_vec3("small_tip_force")[2])) >= self.contact_threshold)
        contacts.thumb = bool(abs(float(self._sensor_vec3("thumb_tip_force")[2])) >= self.contact_threshold)

        self.contact_pub.publish(contacts)

    """
    Main simulation loop running in a separate thread.
    Steps the MuJoCo physics simulation, writes actuator commands, 
    syncs the viewer, and publishes state continuously until the node shuts down.
    """
    def run_simulation(self):
        while self.sim_active:
            try:
                if not rclpy.ok():
                    break

                if self.viewer is None or not self.viewer.is_running():
                    break

                self.write_actuators()
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()

                if not self.sim_active or not rclpy.ok():
                    break

                self.publish_state()

            except Exception as exc:
                msg = str(exc)
                if "destruction was requested" in msg or "publisher's context is invalid" in msg:
                    break

                try:
                    if rclpy.ok():
                        self.get_logger().error(f"Simulation loop error: {exc}")
                except Exception:
                    pass
                break

        self.sim_active = False

    def destroy_node(self):
        self.sim_active = False

        try:
            if hasattr(self, "sim_thread") and self.sim_thread.is_alive():
                self.sim_thread.join(timeout=2.0)
        except Exception:
            pass

        try:
            if hasattr(self, "viewer") and self.viewer is not None:
                self.viewer.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    node = None

    try:
        node = RH8DMujocoNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass

        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()