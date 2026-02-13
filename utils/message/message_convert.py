import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import CompressedImage, JointState
import cv2
from builtin_interfaces.msg import Time
import time
import base64
import cv2 as cv
from utils.message.datatype import RobotAction, Trajectory, ExecutionMode, JointAction, EEAction
from utils.torch_utils import dict_apply
import torch

def header_stamp_to_timestamp(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def timestamp_to_header_stamp(timestamp: float):
    stamp = Time()
    stamp.sec = int(timestamp)
    stamp.nanosec = int((timestamp - stamp.sec) * 1_000_000_000)
    return stamp

def get_action_time(action: RobotAction):
    return header_stamp_to_timestamp(action.left_gripper.header.stamp)

def pose_to_7d_array(pose: Pose):
    return np.array([
        pose.position.x, 
        pose.position.y, 
        pose.position.z, 
        pose.orientation.x, 
        pose.orientation.y, 
        pose.orientation.z, 
        pose.orientation.w
    ], dtype=np.float32)

def compressed_image_to_rgb_array(buffer: np.uint8):
    rgb_np = np.frombuffer(buffer, np.uint8)
    # 统一读取为三通道并转换为 RGB，以匹配训练期
    rgb_bgr = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        return
    res = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return res.transpose(2, 0, 1)

def array_to_joint_state(array: np.ndarray, timestamp: float = None):
    joint_state = JointState()
    if timestamp is None:
        timestamp = time.time()
    joint_state.header.stamp = timestamp_to_header_stamp(timestamp)
    for data in array:
        joint_state.position.append(data)
    return joint_state

def array_to_pose_stamped(array: np.ndarray, timestamp: float = None):
    """
    将 7D 数组 [x, y, z, qx, qy, qz, qw] 转换为 PoseStamped 消息
    """
    pose_stamped = PoseStamped()
    if timestamp is None:
        timestamp = time.time()
    pose_stamped.header.stamp = timestamp_to_header_stamp(timestamp)
    
    if len(array) >= 3:
        pose_stamped.pose.position.x = float(array[0])
        pose_stamped.pose.position.y = float(array[1])
        pose_stamped.pose.position.z = float(array[2])
    
    if len(array) >= 7:
        pose_stamped.pose.orientation.x = float(array[3])
        pose_stamped.pose.orientation.y = float(array[4])
        pose_stamped.pose.orientation.z = float(array[5])
        pose_stamped.pose.orientation.w = float(array[6])
    
    return pose_stamped

def decode_img_from_base64(img_base64: str, output_format="rgb") -> np.ndarray:
    img_data = base64.b64decode(img_base64)
    # 将二进制数据转换为 numpy 数组
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    # 使用 cv2.imdecode 将其恢复为图像
    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if output_format == "rgb":
        return cv2.cvtColor(img_array, cv.COLOR_BGR2RGB)
    else:
        return img_array

def array_to_action_dict(actions_array, execution_mode: ExecutionMode):
    if execution_mode == ExecutionMode.EE_POSE:
        return {
            "left_ee_pose": actions_array[:, :7],
            "right_ee_pose": actions_array[:, 7:14],
        }
    elif execution_mode == ExecutionMode.JOINT_STATE:
        return {
            "left_arm": actions_array[0:6],
            "left_gripper": actions_array[6],
            "right_arm": actions_array[7:13],
            "right_gripper": actions_array[13],
        }

def actions_dict_to_array(actions:dict, execution_mode: ExecutionMode) -> np.ndarray:
    actions_list = []
    actions_keys = actions.keys()
    if execution_mode == ExecutionMode.JOINT_STATE:
        actions_list.append(actions["left_arm"])
        actions_list.append(actions["left_gripper"])
        actions_list.append(actions["right_arm"])
        actions_list.append(actions["right_gripper"])
    elif execution_mode == ExecutionMode.EE_POSE:
        actions_list.append(actions["left_ee_pose"])
        actions_list.append(actions["left_gripper"])
        actions_list.append(actions["right_ee_pose"])
        actions_list.append(actions["right_gripper"])

    if "torso" in actions_keys and actions["torso"] is not None:
        actions_list.append(actions["torso"])

    return np.concatenate(actions_list, axis=2)   

def array_to_action(actions: np.ndarray, execution_mode: ExecutionMode):

    action_kwargs = {}

    if execution_mode == ExecutionMode.EE_POSE:
        action_kwargs["left_ee_pose"] = array_to_pose_stamped(actions[0:7])
        action_kwargs["left_gripper"] = array_to_joint_state(actions[7:8])
        action_kwargs["right_ee_pose"] = array_to_pose_stamped(actions[8:15])
        action_kwargs["right_gripper"] = array_to_joint_state(actions[15:16])
        action_kwargs["torso"] = array_to_joint_state(actions[16:20])
    elif execution_mode == ExecutionMode.JOINT_STATE:
        action_kwargs["left_arm"] = array_to_joint_state(actions[0:6])
        action_kwargs["left_gripper"] = array_to_joint_state(actions[6:7])
        action_kwargs["right_arm"] = array_to_joint_state(actions[7:13])
        action_kwargs["right_gripper"] = array_to_joint_state(actions[13:14])

    return RobotAction(**action_kwargs)


def actions_dict_to_trajectory(actions: dict, time_step: float=0.0666, num_of_steps: int=32, timestamp: float=None) -> Trajectory:
    actions = dict_apply(actions, lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
    
    field_configs = [
        ("left_arm", array_to_joint_state, False),
        ("right_arm", array_to_joint_state, False),
        ("torso", array_to_joint_state, False),
        ("left_gripper", array_to_joint_state, False),
        ("right_gripper", array_to_joint_state, False),
        ("chassis", array_to_joint_state, False),
        ("left_ee_pose", array_to_pose_stamped, True),
        ("right_ee_pose", array_to_pose_stamped, True),
    ]
    
    field_data = {}
    for key, converter, _ in field_configs:
        if key in actions:
            field_data[key] = actions[key][0]

    trajectory = Trajectory()
    trajectory.timestamp = timestamp if timestamp is not None else time.time()
    
    for i in range(num_of_steps):
        action_timestamp = trajectory.timestamp + i * time_step
        action_kwargs = {
        }
        
        for key, converter, _ in field_configs:
            if key in field_data:
                action_kwargs[key] = converter(field_data[key][i], action_timestamp)
            else:
                action_kwargs[key] = None
        
        action = RobotAction(**action_kwargs)
        trajectory.actions.append(action)
    
    return trajectory
