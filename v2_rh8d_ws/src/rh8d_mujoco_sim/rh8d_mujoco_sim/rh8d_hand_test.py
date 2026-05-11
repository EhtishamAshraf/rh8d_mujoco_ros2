#!/usr/bin/env python3

"""
RH8DGestureController: 
    ROS2 node for controlling the RH8D robotic hand.
    Executes predefined gestures while monitoring contacts and tendon forces.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException

from rh8d_mujoco_interfaces.msg import HandCommand, HandState, HandContacts


class RH8DGestureController(Node):
    def __init__(self):
        super().__init__('rh8d_object_pick')

        # Default Quality-of-Service for topics (BEST_EFFORT, non-persistent)
        self.qos_default = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.current_state = HandState()
        self.current_contacts = HandContacts()
        self.have_state = False
        self.have_contacts = False

        # Subscribe to hand state and contact topics
        self.create_subscription(
            HandState,
            '/rh8d/state',
            self.state_callback,
            self.qos_default
        )
        self.create_subscription(
            HandContacts,
            '/rh8d/contacts',
            self.contacts_callback,
            self.qos_default
        )
        
        # Publish hand commands to '/rh8d/command'
        self.command_pub = self.create_publisher(
            HandCommand,
            '/rh8d/command',
            self.qos_default
        )

        # Stable contact detection
        self.contact_counter = {
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'small': 0,
        }
        self.required_contact_cycles = 5

        # Rangefinder gating
        self.range_threshold = 0.25

        # Close values
        self.thumb_flexion = 1.57
        self.finger_flexion = 1.57

        # Per-actuator command rates
        self.rate_forearm = 1.0
        self.rate_palm_axis = 1.0
        self.rate_palm_l = 1.0
        self.rate_thumb_axis = 1.2

        self.rate_index = 0.5
        self.rate_middle = 0.5
        self.rate_thumb = 0.5
        self.rate_ring_small = 0.5

        # Current commanded values
        self.current_command = HandCommand()
        self.current_command.forearm = 0.0
        self.current_command.palm_axis = 0.0
        self.current_command.palm_l = 0.0
        self.current_command.thumb_axis = 0.0
        self.current_command.thumb_tendon = 0.0
        self.current_command.index_tendon = 0.0
        self.current_command.middle_tendon = 0.0
        self.current_command.ring_small_tendon = 0.0

        # Latch tendon commands after first stable grasp contact
        self.grasp_locked = False
        self.latched_tendon_command = {
            'thumb': 0.0,
            'index': 0.0,
            'middle': 0.0,
            'ring_small': 0.0,
        }
        
        # List of predefined gestures
        # Each gesture: target joint positions + optional checks or contact requirements
        # Position order: [forearm, palm_axis, palm_l, thumb_axis, thumb, index, middle, ring_small]
        self.gestures = [
            {
                "name": "Hand Open with Thumb Adduction",
                "positions": [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0],
                "checks": [
                    ('thumb_axis_joint', 1.57),
                ]
            },
            {
                "name": "Hand Close",
                "positions": [0.0, 0.0, 0.0, 1.57, self.thumb_flexion, self.finger_flexion, self.finger_flexion, self.finger_flexion],
                "require_object": True,
                "use_contact": True
            },
            {
                "name": "Wrist Rotation (-)",
                "positions": [-1.57, 0.0, 0.0, 1.57, self.thumb_flexion, self.finger_flexion, self.finger_flexion, self.finger_flexion],
                "checks": [
                    ('forearm_joint', -1.57),
                ]
            },
            # comment out "Wrist Adduction (-)" for the ball
            # {
            #     "name": "Wrist Adduction (-)",
            #     "positions": [-1.57, -0.8, 0.0, 1.57, self.thumb_flexion, self.finger_flexion, self.finger_flexion, self.finger_flexion],
            #     "checks": [
            #         ('palm_axis_joint', -0.8),
            #     ]
            # },
            {
                "name": "Wrist Adduction (+)",
                "positions": [-1.57, 0.8, 0.0, 1.57, self.thumb_flexion, self.finger_flexion, self.finger_flexion, self.finger_flexion],
                "checks": [
                    ('palm_axis_joint', 0.8),
                ]
            },
            {
                "name": "Wrist Flexion (-)",
                "positions": [-1.57, 0.8, -0.6, 1.57, self.thumb_flexion, self.finger_flexion, self.finger_flexion, self.finger_flexion],
                "checks": [
                    ('palm_l_joint', -0.6),
                ]
            },
            {
                "name": "Wrist Rotation (+)",
                "positions": [0.0, 0.0, 0.0, 1.57, self.thumb_flexion, self.finger_flexion, self.finger_flexion, self.finger_flexion],
                "checks": [
                    ('forearm_joint', 0.0),
                ]
            },
            {
                "name": "Hand Open",
                "positions": [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0],
                "checks": [
                    ('thumb_mcp', 0.0),
                    ('index_mcp', 0.0),
                    ('middle_mcp', 0.0),
                    ('ring_mcp', 0.0),
                ]
            }
        ]

        self.state_index = 0
        self.state_sent = False
        self.start_time = self.get_clock().now()

        # Timer to periodically run the step() function (50ms)
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.step)

        self.get_logger().info("RH8DGestureController started...")

    # Update internal state when HandState messages are received
    def state_callback(self, msg: HandState):
        self.current_state = msg
        self.have_state = True

    # Update internal contact info when HandContacts messages are received
    def contacts_callback(self, msg: HandContacts):
        self.current_contacts = msg
        self.have_contacts = True

    # Check if object is within range
    def object_present(self) -> bool:
        rf = self.current_state.palm_range
        return (rf > 0.0) and (rf < self.range_threshold)

    # Publishing commands
    def publish_command(self, cmd: HandCommand):
        self.command_pub.publish(cmd)

    # Move a current value towards a target value at a given rate and timestep (dt)
    def move_towards(self, current: float, target: float, rate: float, dt: float) -> float:
        if current < target:
            return min(current + rate * dt, target)
        elif current > target:
            return max(current - rate * dt, target)
        return current

    # Gradually move all hand joints towards the target positions
    # Returns True if all joints have reached the target positions
    def ramp_command_towards(self, target_positions, dt):
        
        new_cmd = HandCommand()

        new_cmd.forearm = self.move_towards(
            self.current_command.forearm, target_positions[0], self.rate_forearm, dt
        )
        new_cmd.palm_axis = self.move_towards(
            self.current_command.palm_axis, target_positions[1], self.rate_palm_axis, dt
        )
        new_cmd.palm_l = self.move_towards(
            self.current_command.palm_l, target_positions[2], self.rate_palm_l, dt
        )
        new_cmd.thumb_axis = self.move_towards(
            self.current_command.thumb_axis, target_positions[3], self.rate_thumb_axis, dt
        )

        new_cmd.thumb_tendon = self.move_towards(
            self.current_command.thumb_tendon, target_positions[4], self.rate_thumb, dt
        )
        new_cmd.index_tendon = self.move_towards(
            self.current_command.index_tendon, target_positions[5], self.rate_index, dt
        )
        new_cmd.middle_tendon = self.move_towards(
            self.current_command.middle_tendon, target_positions[6], self.rate_middle, dt
        )
        new_cmd.ring_small_tendon = self.move_towards(
            self.current_command.ring_small_tendon, target_positions[7], self.rate_ring_small, dt
        )

        self.current_command = new_cmd
        self.publish_command(new_cmd)

        reached = (
            abs(new_cmd.forearm - target_positions[0]) < 1e-6 and
            abs(new_cmd.palm_axis - target_positions[1]) < 1e-6 and
            abs(new_cmd.palm_l - target_positions[2]) < 1e-6 and
            abs(new_cmd.thumb_axis - target_positions[3]) < 1e-6 and
            abs(new_cmd.thumb_tendon - target_positions[4]) < 1e-6 and
            abs(new_cmd.index_tendon - target_positions[5]) < 1e-6 and
            abs(new_cmd.middle_tendon - target_positions[6]) < 1e-6 and
            abs(new_cmd.ring_small_tendon - target_positions[7]) < 1e-6
        )

        return reached

    # Get the current value of a joint field by name
    def get_state_value(self, field_name: str) -> float:
        return getattr(self.current_state, field_name)

    # Check if a specific joint value has reached a target within a tolerance
    def is_value_at_target(self, field_name: str, target: float, tolerance: float = 0.03) -> bool:
        current = self.get_state_value(field_name)
        diff = abs(current - target)
        self.get_logger().info(
            f"difference for {field_name}: {diff:.4f} (current: {current:.4f}, target: {target:.4f})"
        )
        return diff <= tolerance

    # Check if a finger has maintained contact for a required number of cycles
    def is_grasp_contact_stable(self, finger_name: str) -> bool:
        contact_now = getattr(self.current_contacts, finger_name)

        if contact_now:
            self.contact_counter[finger_name] += 1
        else:
            self.contact_counter[finger_name] = 0

        return self.contact_counter[finger_name] >= self.required_contact_cycles

    # Latch the current tendon values to maintain the grasp
    def latch_current_grasp(self):
        self.latched_tendon_command['thumb'] = self.current_command.thumb_tendon
        self.latched_tendon_command['index'] = self.current_command.index_tendon
        self.latched_tendon_command['middle'] = self.current_command.middle_tendon
        self.latched_tendon_command['ring_small'] = self.current_command.ring_small_tendon
        self.grasp_locked = True

        self.get_logger().info(
            f"Latched grasp | "
            f"thumb={self.latched_tendon_command['thumb']:.4f}, "
            f"index={self.latched_tendon_command['index']:.4f}, "
            f"middle={self.latched_tendon_command['middle']:.4f}, "
            f"ring_small={self.latched_tendon_command['ring_small']:.4f}"
        )

    # Apply latched tendon commands to the target positions if grasp is locked
    # Ensures the hand keeps holding the object even if higher-level commands change
    def apply_latched_tendons(self, positions):
        new_positions = list(positions)

        if self.grasp_locked:
            new_positions[4] = self.latched_tendon_command['thumb']
            new_positions[5] = self.latched_tendon_command['index']
            new_positions[6] = self.latched_tendon_command['middle']
            new_positions[7] = self.latched_tendon_command['ring_small']

        return new_positions

    # Main control loop called periodically by the timer
    # Handles gesture sequencing, object detection, contact-based grasping, and command publishing
    def step(self):
        if not self.have_state or not self.have_contacts:
            return

        if (self.get_clock().now() - self.start_time).nanoseconds < 1e9:
            return

        if self.state_index >= len(self.gestures):
            self.get_logger().info("All gestures completed.")
            if rclpy.ok():
                rclpy.shutdown()
            return

        gesture = self.gestures[self.state_index]

        if gesture["name"] in ["Hand Open with Thumb Adduction", "Hand Open"]:
            self.grasp_locked = False

        # If this gesture requires object presence, do not close until object is detected
        if gesture.get("require_object", False) and not self.object_present():
            self.get_logger().info(
                f"Object not present | palm_range={self.current_state.palm_range:.4f}. Fingers will remain open."
            )
            # Hold current open/adducted posture instead of closing
            safe_open = [0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0]
            self.ramp_command_towards(safe_open, self.timer_period)
            return

        if not self.state_sent:
            self.get_logger().info(f"Gesture {self.state_index}: {gesture['name']}")
            self.state_sent = True
            self.contact_counter = {
                'thumb': 0,
                'index': 0,
                'middle': 0,
                'ring': 0,
                'small': 0,
            }

        target_positions = gesture["positions"]

        if gesture.get("use_contact", False):
            self.get_logger().info(
                f"Contacts | thumb={self.current_contacts.thumb}, "
                f"index={self.current_contacts.index}, "
                f"middle={self.current_contacts.middle}, "
                f"ring={self.current_contacts.ring}, "
                f"small={self.current_contacts.small}, "
                f"palm_range={self.current_state.palm_range:.4f}"
            )

            contact_done = all([
                self.is_grasp_contact_stable('thumb'),
                self.is_grasp_contact_stable('middle'),
                self.is_grasp_contact_stable('ring') or self.is_grasp_contact_stable('small'),
            ])

            if contact_done and not self.grasp_locked:
                self.latch_current_grasp()

            if self.grasp_locked:
                target_positions = self.apply_latched_tendons(target_positions)

            self.ramp_command_towards(target_positions, self.timer_period)

            # Allow either stable contact OR fully reached closed pose
            done = self.grasp_locked

        # Normal gestures without contact-based locking
        else:
            target_positions = self.apply_latched_tendons(target_positions)
            self.ramp_command_towards(target_positions, self.timer_period)

            done = all(
                self.is_value_at_target(field_name, target)
                for field_name, target in gesture["checks"]
            )

        if done:
            self.get_logger().info(f"{gesture['name']} complete.")
            self.state_index += 1
            self.state_sent = False


def main(args=None):
    rclpy.init(args=args)
    node = RH8DGestureController()
    ex = SingleThreadedExecutor()
    ex.add_node(node)

    try:
        while rclpy.ok():
            ex.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            ex.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()