from typing import Dict, Literal
from dataclasses import dataclass, field
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped

@dataclass(frozen=True)
class Topic:
    channel: str
    msg_type: CompressedImage | JointState | PoseStamped

@dataclass
class RobotTopicsConfig:
    state: Dict[str, Topic] = field(
        default_factory=lambda: {
            "left_arm": Topic("/hdas/feedback_arm_left", JointState),
            "right_arm": Topic("/hdas/feedback_arm_right", JointState),
            "torso": Topic("/hdas/feedback_torso", JointState),
            "chassis": Topic("/hdas/feedback_chassis", JointState),
            "left_ee_pose": Topic("/motion_control/pose_ee_arm_left", PoseStamped),
            "right_ee_pose": Topic("/motion_control/pose_ee_arm_right", PoseStamped),
            "left_gripper": Topic("/hdas/feedback_gripper_left", JointState),
            "right_gripper": Topic("/hdas/feedback_gripper_right", JointState),
        }
    )

    images: Dict[str, Topic] = field(
        default_factory=lambda: {
            "head_rgb": Topic("/hdas/camera_head/left_raw/image_raw_color/compressed", CompressedImage),
            "left_wrist_rgb": Topic("/hdas/camera_wrist_left/color/image_raw/compressed", CompressedImage),
            "right_wrist_rgb": Topic("/hdas/camera_wrist_right/color/image_raw/compressed", CompressedImage),
        }
    )

    action: Dict[str, Topic] = field(
        default_factory=lambda: {
            "left_arm": Topic("/motion_target/target_joint_state_arm_left", JointState),
            "right_arm": Topic("/motion_target/target_joint_state_arm_right", JointState),
            "torso": Topic("/motion_target/target_joint_state_torso", JointState),
            "left_ee_pose": Topic("/motion_target/target_pose_arm_left", PoseStamped),
            "right_ee_pose": Topic("/motion_target/target_pose_arm_right", PoseStamped),
            "left_gripper": Topic("/motion_target/target_position_gripper_left", JointState),
            "right_gripper": Topic("/motion_target/target_position_gripper_right", JointState),
        }
    )

    qos: Dict[str, QoSProfile] = field(
        default_factory=lambda: {
            "sub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
            ),
            "pub": QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.VOLATILE
            ),
        }
    )

    camera_deque_length: int = 3 # 15Hz, for 0.2s
    state_deque_length: int = 80 # >400 Hz, for 0.2s

