import argparse
import yaml
from enum import Enum
from loguru import logger
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
import time
from teleoperation_msg_ros2.srv import SwitchControlModeVR

DEFAULT_QOS_PUB = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

class ResetType(Enum):
    QUIT = 0
    ALL = 1
    LEFT_ARM = 2
    RIGHT_ARM = 3
    LEFT_GRIPPER = 4
    RIGHT_GRIPPER = 5
    TORSO = 6

class ResetHelper:
    def __init__(self, args):
        yaml_file = args.yaml_file
        self.duration = 1.0
        self.node = rclpy.create_node('reset_helper')
        with open(yaml_file, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.init_pose_info = self._search_init_pose(self.config)[0]["value"]
        logger.info(f"init_pose_info: {self.init_pose_info}")
        self.target_joint_state_pub_left = self.node.create_publisher(
            JointState, '/motion_target/target_joint_state_arm_left', DEFAULT_QOS_PUB)
        self.target_joint_state_pub_right = self.node.create_publisher(
            JointState, '/motion_target/target_joint_state_arm_right', DEFAULT_QOS_PUB)
        self.gripper_pub_left = self.node.create_publisher(
            JointState, '/motion_target/target_position_gripper_left', DEFAULT_QOS_PUB)
        self.gripper_pub_right = self.node.create_publisher(
            JointState, '/motion_target/target_position_gripper_right', DEFAULT_QOS_PUB)
        self.torso_pub = self.node.create_publisher(
            JointState, '/motion_target/target_joint_state_torso', DEFAULT_QOS_PUB)
        self.client = self.node.create_client(
            SwitchControlModeVR, '/switch_control_mode_vr')
    def start(self):
        right_arm_pose = self.init_pose_info["right_arm_init_position"]
        left_arm_pose = self.init_pose_info["left_arm_init_position"]
        left_gripper_pose = self.init_pose_info["left_gripper_position"]
        right_gripper_pose = self.init_pose_info["right_gripper_position"]
        torso_pose = self.init_pose_info["torso_position"]

        while True:
            reset_type = self._get_keyboard_input()
            match reset_type:
                case ResetType.QUIT:
                    logger.info("quit")
                    exit(0)
                case ResetType.ALL:

                    logger.info("reset all")
                    self.send_reset_request()
                    for _ in range(300):
                        self.target_joint_state_pub_left.publish(JointState(position=left_arm_pose))
                        self.target_joint_state_pub_right.publish(JointState(position=right_arm_pose))
                        self.gripper_pub_left.publish(JointState(position=[left_gripper_pose]))
                        self.gripper_pub_right.publish(JointState(position=[right_gripper_pose]))
                        self.torso_pub.publish(JointState(position=torso_pose))
                        time.sleep(0.01)
                    continue
                case ResetType.LEFT_ARM:
                    logger.info(f"reset left_arm to {left_arm_pose}")
                    for _ in range(100):
                        self.target_joint_state_pub_left.publish(JointState(position=left_arm_pose))
                        time.sleep(0.01)
                    continue
                case ResetType.RIGHT_ARM:
                    logger.info(f"reset right_arm to {right_arm_pose}")
                    for _ in range(100):
                        self.target_joint_state_pub_right.publish(JointState(position=right_arm_pose))
                        time.sleep(0.01)
                    continue
                case ResetType.LEFT_GRIPPER:
                    logger.info(f"reset left_gripper to {left_gripper_pose}")
                    for _ in range(100):
                        self.gripper_pub_left.publish(JointState(position=[left_gripper_pose]))
                        time.sleep(0.01)
                    continue
                case ResetType.RIGHT_GRIPPER:
                    logger.info(f"reset right_gripper to {right_gripper_pose}")
                    for _ in range(100):
                        self.gripper_pub_right.publish(JointState(position=[right_gripper_pose]))
                        time.sleep(0.01)
                    continue
                case ResetType.TORSO:
                    logger.info(f"reset torso to {torso_pose}")
                    for _ in range(100):
                        self.torso_pub.publish(JointState(position=torso_pose))
                        time.sleep(0.01)
                    continue

    def _get_keyboard_input(self):
        logger.info(f'Press keys to reset')
        logger.info("0: quit")
        logger.info("1: reset all")
        logger.info("2: reset left_arm")
        logger.info("3: reset right_arm")
        logger.info("4: reset left_gripper")
        logger.info("5: reset right_gripper")
        logger.info("6: reset torso")
        key = int(input())
        return ResetType(key)

    def reset(self):
        pass

    def _search_init_pose(self, yaml_data, current_path=""):
        results = []
        if isinstance(yaml_data, dict):
            for key, value in yaml_data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                # 如果找到init_pose字段
                if key == "init_pose":
                    results.append({
                        "path": new_path,
                        "value": value
                    })
                
                # 递归查找子元素
                results.extend(self._search_init_pose(value, new_path))
        
        elif isinstance(yaml_data, list):
            for i, item in enumerate(yaml_data):
                new_path = f"{current_path}[{i}]"
                results.extend(self._search_init_pose(item, new_path))
        
        return results

    def send_reset_request(self):
        # 设置请求中的use_vr_mode参数为False
        # self.get_logger().info("send reset service request")
        req = SwitchControlModeVR.Request()
        req.use_vr_mode = False
        # 发送异步请求
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        logger.info("vr模式切换和重置成功")
        return future
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    rclpy.init()
    reset_helper = ResetHelper(args)
    reset_helper.start()
    rclpy.shutdown()
