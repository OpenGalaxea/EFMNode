import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from functools import partial
from loguru import logger
import torch
import numpy as np
import time
import threading
from dataclasses import asdict

from core.communication.robot_topics import RobotTopicsConfig
from core.communication.message_queue import MessageQueue

from utils.message.message_convert import (
    header_stamp_to_timestamp,
    pose_to_7d_array,
    compressed_image_to_rgb_array,
    array_to_joint_state
)
from utils.message.datatype import RobotAction

class Ros2Bridge:
    def __init__(self, config, cfg, use_recv_time: bool = False, num_threads: int = None):
        self.topics_config = RobotTopicsConfig()
        self.use_recv_time = use_recv_time
        self.cfg = cfg
        self.hardware = config["robot"]["hardware"]
        self.enable_publish = config["robot"]["enable_publish"]
        logger.info(f"config.toml:\nhardware:{self.hardware}\nenable publish: {self.enable_publish}")
        
        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node("ros2_bridge")
        self.subscribers = {}
        self.publishers = {}

        if config["robot"]["hardware"] == "R1_LITE":
            self.dof_of_arm = 6
        elif config["robot"]["hardware"] == "R1_PRO":
            self.dof_of_arm = 7

        self.obs_buffer = {}
        
        self.callback_group = ReentrantCallbackGroup()
        
        self.executor = MultiThreadedExecutor(num_threads=num_threads)
        self.last_obs_time = None
        
        self.init_topics()

        self.executor.add_node(self.node)
        

        import threading
        self._executor_thread = threading.Thread(target=self._run_executor, daemon=True)
        self._executor_thread.start()
        logger.info(f"Ros2Bridge init.")

    def init_topics(self):
        for name, topic in self.topics_config.images.items():
            channel, msg_type = topic.channel, topic.msg_type
            self.obs_buffer[name] = MessageQueue(maxlen=self.topics_config.camera_deque_length)
            self.subscribers[channel] = self.node.create_subscription(
                msg_type, 
                channel, 
                partial(self.image_callback, _stack=self.obs_buffer[name]), 
                self.topics_config.qos["sub"],
                callback_group=self.callback_group
            )

        for name, topic in self.topics_config.state.items():
            channel, msg_type = topic.channel, topic.msg_type
            self.obs_buffer[name] = MessageQueue(maxlen=self.topics_config.state_deque_length)

            if "ee_pose" in name:
                callback = partial(self.pose_callback, _stack=self.obs_buffer[name])
            else:
                callback = partial(self.state_callback, _stack=self.obs_buffer[name], state_name=name)

            self.subscribers[channel] = self.node.create_subscription(
                msg_type, 
                channel, 
                callback,
                self.topics_config.qos["sub"],
                callback_group=self.callback_group\
            )

        for name, topic in self.topics_config.action.items():
            channel, msg_type = topic.channel, topic.msg_type
            self.publishers[topic] = self.node.create_publisher(
                msg_type, 
                channel, 
                self.topics_config.qos["pub"]
            )

    def is_running(self):
        return rclpy.ok()

    def publish_action(self, action: RobotAction):
        for name, msg in asdict(action).items():
            if msg is not None and name in self.enable_publish:
                self.publishers[self.topics_config.action[name]].publish(msg)

    def reset(self, step_size=0.2, freq = 5):
        # reset to zero joints and close grippers
        logger.info("resetting")
        while True:
            joint_feedback_topics = [topic for topic in self.topics_config.state.keys() if 'ee' not in topic]
            action = {}
            for name in joint_feedback_topics:
                if "gripper" in name:
                    # close grippers
                    # action[name] = np.array([0])
                    action[name] = [0.]
                elif "chassis" in name or "torso" in name:
                    continue
                else:
                    latest_feedback = np.array(self.obs_buffer[name][-1]['data'][:6])
                    step_size_with_dir = step_size * np.sign(-latest_feedback)
                    step_size_with_dir = np.where(np.abs(latest_feedback) < step_size,
                                                  -latest_feedback, step_size_with_dir)
                    motion_target = latest_feedback + step_size_with_dir
                    action[name] = motion_target.tolist()

            for k, v in action.items():
                msg = array_to_joint_state(v)
                self.publishers[self.topics_config.action[k]].publish(msg)
            if all([np.all(np.array(value) == 0) for _, value in action.items()]):
                break
            time.sleep(1 / freq)

    def register_publish_callback(self, frequency: float, callback: callable):
        self.timer = self.node.create_timer(1.0 / frequency, callback)

    def register_subscription(self, message_type: type, topic: str, callback: callable):
        self.subscribers[topic] = self.node.create_subscription(
            message_type,
            topic,
            callback,
            self.topics_config.qos["sub"],
            callback_group=self.callback_group
        )


    def _find_nearest_message(self, buffer: MessageQueue, target_time: float) -> dict:
        if len(buffer) == 0:
            return None
        
        best_msg = None
        best_diff = float('inf')
        
        for i in range(len(buffer) - 1, -1, -1):
            msg = buffer[i]
            time_diff = abs(msg["message_time"] - target_time)
            if time_diff < best_diff:
                best_diff = time_diff
                best_msg = msg
                if time_diff < 0.001:
                    break
        
        return best_msg
    
    def gather_obs(self, device: torch.device = torch.device("cuda")):
        head_rgb_key = "head_rgb"
        if head_rgb_key not in self.obs_buffer or len(self.obs_buffer[head_rgb_key]) == 0:
            logger.warning("Head camera buffer is empty")
            return None, None
        
        head_buffer = self.obs_buffer[head_rgb_key]
        head_msg = head_buffer[-1]
        reference_time = head_msg["message_time"]

        if self.last_obs_time == reference_time:
            logger.warning(f'No new message in Head camera buffer')
            return None, None
        
        obs = {"images": {}, "state": {}}
        
        for name, buffer in self.obs_buffer.items():
            if len(buffer) == 0:
                logger.warning(f"Buffer {name} is empty, skipping")
                return None, None
            
            if name == head_rgb_key:
                data = head_msg["data"]
            else:
                nearest_msg = self._find_nearest_message(buffer, reference_time)
                if nearest_msg is None:
                    logger.warning(f"Failed to find nearest message for {name}")
                    return None, None
                data = nearest_msg["data"]
            
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data).unsqueeze(0) if hasattr(data, '__array__') else torch.tensor(data)
            
            if name in self.topics_config.images:
                obs["images"][name] = data.to(device)#.unsqueeze(0)
            else:
                obs["state"][name] = data.float()
                
            
            if name == 'chassis':
                obs["state"][name] = obs["state"][name][..., 0: 3]
                torch.atan2(torch.sin(obs["state"][name]), torch.cos(obs["state"][name]))

        obs["state_is_pad"] = torch.tensor([False])
        obs["image_is_pad"] = torch.tensor([False])
        obs["action_is_pad"] = torch.tensor([False]*50)
        obs["idx"] = torch.tensor(0)

        self.last_obs_time = reference_time

        return reference_time, obs


    def _run_executor(self):
        """执行器运行方法（在独立线程中调用）"""
        try:
            self.executor.spin()
        except Exception as e:
            logger.error(f"Executor error: {e}")
        finally:
            logger.info("Executor stopped")
    
    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9
    
    def destroy(self):
        self.executor.shutdown()
        
        if self._executor_thread and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=2.0)
            if self._executor_thread.is_alive():
                logger.warning("Executor thread did not stop in time")

        for subscriber in self.subscribers.values():
            subscriber.destroy()
        for publisher in self.publishers.values():
            publisher.destroy()
        
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()
        
        logger.info("ROS2Bridge destroyed")

    def _create_data_dict(self, timestamp, data):
        if self.use_recv_time:
            return {
                "message_time": self.now(),
                "data": data,
                "receive_time": self.now(),
                "header_time": timestamp,
            }
        else:
            return {
                "message_time": timestamp,
                "data": data,
                "receive_time": self.now(),
                "header_time": timestamp,
            }
    
    def image_callback(self, msg: CompressedImage, _stack=None):
        data_dict = self._create_data_dict(
            timestamp=header_stamp_to_timestamp(msg.header.stamp),
            data=compressed_image_to_rgb_array(msg.data))
        _stack.append(data_dict)

    def state_callback(self, msg: JointState, _stack=None, state_name: str="None"):
        if state_name == "chassis":
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.velocity))
            _stack.append(data_dict)
        elif state_name == "torso":
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position))
            _stack.append(data_dict)
        elif "arm" in state_name:
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position[:self.dof_of_arm]))
            _stack.append(data_dict)
        elif "gripper" in state_name:
            data_dict = self._create_data_dict(
                timestamp=header_stamp_to_timestamp(msg.header.stamp),
                data=np.array(msg.position))
            _stack.append(data_dict)
        else:
            raise ValueError(f"Invalid state name: {state_name}")

    def pose_callback(self, msg: PoseStamped, _stack=None):
        data_dict = self._create_data_dict(
            timestamp=header_stamp_to_timestamp(msg.header.stamp),
            data=pose_to_7d_array(msg.pose))
    
        _stack.append(data_dict)

