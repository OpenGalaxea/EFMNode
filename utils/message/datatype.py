from dataclasses import dataclass, field
from typing import Optional, Literal
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from collections import deque
import time
from enum import Enum
import numpy as np

@dataclass
class RobotAction:
    left_arm: Optional[JointState] = None
    right_arm: Optional[JointState] = None
    torso: Optional[JointState] = None
    left_gripper: Optional[JointState] = None
    right_gripper: Optional[JointState] = None
    chassis: Optional[JointState] = None
    left_ee_pose: Optional[PoseStamped] = None
    right_ee_pose: Optional[PoseStamped] = None

@dataclass
class Trajectory:
    timestamp: float = field(default_factory=lambda: time.time())
    actions: deque = field(default_factory=lambda: deque(maxlen=100))

class ExecutionMode(Enum):
    EE_POSE = "EE_POSE"
    JOINT_STATE = "JOINT_STATE"

@dataclass
class EEAction:
    ee_left_trans_dims = [0, 1, 2]
    ee_left_quat_dims = [3, 4, 5, 6]
    grp_left_dims = [7]
    ee_right_trans_dims = [8, 9, 10]
    ee_right_quat_dims = [11, 12, 13, 14]
    grp_right_dims = [15]
    torso_dims = [16, 17, 18, 19]

    def __init__(self, actions, actions_time, idx, mode: Literal["next", "interp"]="next"):
        self.actions = actions
        self.time = actions_time
        self.idx = idx
        self.mode = mode

    def is_within(self, now):
        return now > self.time[0] and now < self.time[-1]
    def get_action(self, now):
        step = np.argmax(self.time > now)
        if self.mode == "next":
            return self.idx, step, self.actions[step]
        elif self.mode == "interp":
            action = np.interp(now, [self.time[step - 1], self.time[step]], [self.actions[step - 1], self.actions[step]])
            return self.idx, step, action
        else:
            raise NotImplementedError


class JointAction:
    left_arm_joint_dims = [0, 1, 2, 3, 4, 5]
    grp_left_dims = [6]
    right_arm_joint_dims = [7, 8, 9, 10, 11, 12]
    grp_right_dims = [13]

    def __init__(self, actions, actions_time, idx, mode: Literal["next", "interp"]="next"):
        self.actions = actions
        self.time = actions_time
        self.idx = idx
        self.mode = mode

    def is_within(self, now):
        return now > self.time[0] and now < self.time[-1]

    def get_action(self, now):
        step = np.argmax(self.time > now)
        if self.mode == "next":
            return self.idx, step, self.actions[step]
        elif self.mode == "interp":
            action = np.interp(now, [self.time[step - 1], self.time[step]], [self.actions[step - 1], self.actions[step]])
            return self.idx, step, action
        else:
            raise NotImplementedError

