from core.inference.factory import create_inference_engine
from core.processor.factory import create_processor
from core.communication.ros2_bridge import Ros2Bridge
from utils.message.message_convert import actions_dict_to_trajectory 

from scheduler.instruction.instruction import InstructionManager, InstructionAction
from scheduler.trajectory.manager import TrajectoryManager, EnsembleMode
from utils.message.datatype import Trajectory, ExecutionMode
from std_msgs.msg import String
from loguru import logger

import toml
import torch
import time
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from galaxea_fm.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

from accelerate import PartialState
distributed_state = PartialState()

class Scheduler:
    def __init__(self, config):
        self.schedule_config = config
        self.model_config = OmegaConf.load(f"{self.schedule_config['model']['ckpt_dir']}/config.yaml")

        self.cnt = 0
        self.step_mode = self.schedule_config['basic']['step_mode']
        self.step_freq = self.schedule_config['basic']['control_frequency']
        self.num_of_steps = self.schedule_config['basic']['action_steps']

        self._setup_all()

    def run(self):
        while self.ros2_bridge.is_running():
            obs_time, obs = self.ros2_bridge.gather_obs()
            infer_start = time.time()
            actions = self.inference(obs)
            infer_cost = time.time() - infer_start
            if actions is not None and self.cnt >= 2:
                logger.info(f'Infer cost: {infer_cost}')
                self.step(actions['action'], obs_time)
            self.cnt += 1

    def inference(self, obs):
        if obs is None:
            if self.cnt % 100 == 0:
                logger.info("No observation")
            time.sleep(0.01)
            return

        instruct_action = self.instruction_manager.get_instruction(obs)
        if instruct_action == InstructionAction.RESET:
            self.ros2_bridge.reset()
            return
        elif instruct_action == InstructionAction.CONTINUE:
            pass
        elif instruct_action == InstructionAction.SKIP:
            return

        batch = self.processor.preprocess(obs)
        for k, v in batch.items():
            if isinstance(v, str):
                batch[k] = [v]
            elif isinstance(v, bool):
                batch[k] = torch.tensor([v])
            else:
                batch[k] = v.unsqueeze(0)
        batch = self.inference_engine.predict_action(batch)
        batch["action"] = batch["action"].cpu()
        batch["proprio"] = batch["proprio"].cpu()
        actions = self.processor.postprocess(batch)
        return actions

    def step(self, actions: dict, obs_time: float):
        if self.step_mode == "sync":
            trajectory = actions_dict_to_trajectory(actions=actions, time_step=1/self.step_freq, num_of_steps=self.num_of_steps, timestamp=self.ros2_bridge.now())
            if len(trajectory.actions) < self.num_of_steps:
                raise ValueError(f"Trajectory actions length {len(trajectory.actions)} is less than num_of_steps {self.num_of_steps}")

            self._sync_publish(trajectory)

        elif self.step_mode == "async":
            logger.info(f'Add actions to trajectory manager.')
            self.trajectory_manager.add_actions(actions, obs_time)
        else:
            raise ValueError(f"Invalid step mode: {self.step_mode}")

    def _sync_publish(self, trajectory: Trajectory):
        for i in range(self.num_of_steps):
            self.ros2_bridge.publish_action(trajectory.actions[i])
            time.sleep(1.0 / self.step_freq)

    @logger.catch
    def _async_publish(self):
        if not self.trajectory_manager.is_ready():
            return
        now = time.time()
        action = self.trajectory_manager.get_action(now)
        if action is None:
            return
        self.ros2_bridge.publish_action(action)

    def _setup_all(self):
        self._setup_processor()
        self._setup_trajectory_manager()
        self._setup_instruction_manager()
        self._setup_ros2_bridge()
        self._setup_inference_engine()

    def _setup_inference_engine(self):
        self.inference_engine = create_inference_engine(self.schedule_config, self.model_config, use_trt=self.schedule_config['model']['use_trt'])
        self.inference_engine.load_model()

    def _setup_processor(self):
        self.processor = create_processor(self.schedule_config, self.model_config, processor_type=self.schedule_config['model']['processor'])
        self.processor.initialize(Path(f"{self.schedule_config['model']['ckpt_dir']}/dataset_stats.json"))

    def _setup_trajectory_manager(self):
        if self.schedule_config['trajectory']['ensemble_mode'] == "RTC":
            ensemble_mode = EnsembleMode.RTC
        elif self.schedule_config['trajectory']['ensemble_mode'] == "RTG":
            ensemble_mode = EnsembleMode.RTG
        elif self.schedule_config['trajectory']['ensemble_mode'] == "HATO":
            ensemble_mode = EnsembleMode.HATO
        else:
            logger.warning(f"Invalid ensemble mode:{self.schedule_config['trajectory']['ensemble_mode']}")
            ensemble_mode = EnsembleMode.NONE
        
        if self.schedule_config['trajectory']['execution_mode'] == "JOINT_STATE":
            execution_mode = ExecutionMode.JOINT_STATE
        elif self.schedule_config['trajectory']['execution_mode'] == "EE_POSE":
            execution_mode = ExecutionMode.EE_POSE
        else:
            raise ValueError(f"Invalid execution mode: {self.schedule_config['trajectory']['execution_mode']}")
        
        self.trajectory_manager = TrajectoryManager(ensemble_mode=ensemble_mode, execution_mode=execution_mode)
        self.trajectory_manager.start()

    def _setup_instruction_manager(self):
        self.instruction_manager = InstructionManager(self.schedule_config["instruction"])

    def _setup_ros2_bridge(self):
        # HACK: use_recv_time=True to use the received time from ROS2 messages
        self.ros2_bridge = Ros2Bridge(self.schedule_config, self.model_config, use_recv_time=True)
        self.ros2_bridge.register_subscription(String, 'hs/vlm_out2vla', self.instruction_manager._ehi_instruction_callback)
        
        if self.step_mode == "async":
            self.ros2_bridge.register_publish_callback(self.step_freq, self._async_publish)
 

if __name__ == "__main__":
    config = toml.load("config.toml")
    scheduler = Scheduler(config)
    scheduler.run()
