from core.inference.factory import create_inference_engine
from core.processor.factory import create_processor
from core.communication.ros2_bridge import Ros2Bridge
from utils.message.message_convert import actions_dict_to_trajectory 

from scheduler.instruction.instruction import InstructionManager, InstructionAction
from std_msgs.msg import String

import toml
import time
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split('/')[int(idx)])

from accelerate import PartialState
distributed_state = PartialState()

class Scheduler:
    def __init__(self, config):
        cfg = OmegaConf.load(f"{config['model']['ckpt_dir']}/config.yaml")
        self.inference_engine = create_inference_engine(config, cfg, use_trt=config['model']['use_trt'])
        self.inference_engine.load_model()
        self.processor = create_processor(config, cfg, processor_type=config['model']['processor'])
        self.processor.initialize(Path(f"{config['model']['ckpt_dir']}/dataset_stats.json"))
        self.ros2_bridge = Ros2Bridge(config, cfg)
        self.instruction_manager = InstructionManager(config["instruction"])
        self.ros2_bridge.register_subscription(String, 'hs/vlm_out2vla', self.instruction_manager._ehi_instruction_callback)
        self.cnt = 0
        self.step_mode = config['basic']['step_mode']
        self.step_freq = config['basic']['control_frequency']
        self.num_of_steps = config['basic']['action_steps']

    def run(self):
        while self.ros2_bridge.is_running():
            actions = self.inference()
            if actions is not None:
                self.step(actions['action'])
            self.cnt += 1

    def inference(self):
        obs = self.ros2_bridge.gather_obs()
        if obs is None:
            if self.cnt % 100 == 0:
                print("No observation")
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
            batch[k] = v.unsqueeze(0)
        batch =self.inference_engine.predict_action(batch)
        batch["action"] = batch["action"].cpu()
        batch["proprio"] = batch["proprio"].cpu()
        actions = self.processor.postprocess(batch)
        return actions

    def step(self, actions: dict):
        trajectory = actions_dict_to_trajectory(actions=actions, time_step=1/self.step_freq, num_of_steps=self.num_of_steps, timestamp=self.ros2_bridge.now())
        if self.step_mode == "sync":
            if len(trajectory.actions) < self.num_of_steps:
                raise ValueError(f"Trajectory actions length {len(trajectory.actions)} is less than num_of_steps {self.num_of_steps}")

            for i in range(self.num_of_steps):
                self.ros2_bridge.publish_action(trajectory.actions[i])
                time.sleep(1.0 / self.step_freq)
 
