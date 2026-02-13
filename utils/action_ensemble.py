import time
import numpy as np
from typing import Literal
from scipy.spatial.transform import Rotation as R, Slerp

from .thread_safe_queue import ThreadSafeDeque

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
    left_arm_joint_dims = [0, 1, 2, 3, 4, 5, 6]
    grp_left_dims = [7]
    right_arm_joint_dims = [8, 9, 10, 11, 12, 13, 14]
    grp_right_dims = [15]

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


class ActionEnsemble:
    def __init__(
        self, 
        control_frequency: float, 
        mode: Literal["latest", "avg", "ACT", "HATO", "RTC"], 
        step: int, 
        k_act: float, 
        tau_hato: float, 
        use_fix_quat: bool, 
        use_slerp_quat_avg: bool, 
        logger, 
    ):
        assert mode in ["latest", "avg", "ACT", "HATO", "RTC"]
        self.dt = 1.0 / control_frequency
        self.step = step
        self.mode = mode
        self.k_act = k_act
        self.tau_hato = tau_hato
        self.use_fix_quat = use_fix_quat
        self.use_slerp_quat_avg = use_slerp_quat_avg
        self.logger = logger
        self.action_buffer = ThreadSafeDeque()
        self.logger.info(f"Ensemble mode: {self.mode}")
        self.add_idx = 0
        self.get_idx = 0
        self.last_pub_action = None
    
    def add_action(self, actions, obs_time):
        now = time.time()
        if not isinstance(actions, dict):
            actions = actions[0: self.step]
        else:
            actions = np.concatenate([actions["left_ee_pose"], actions["left_gripper"], actions["right_ee_pose"], actions["right_gripper"], actions["torso"]], axis=2)
            actions = actions[0: self.step]
        actions = actions.squeeze(0)
        actions_time = obs_time + self.dt * np.arange(actions.shape[0])
        mask = actions_time > now
        if not np.any(mask):
            self.logger.warning("The added new action chunk is discarded as it is outdated")
            return self.step

        if self.use_fix_quat:
            actions[:, EEAction.ee_left_quat_dims] = ActionEnsemble.fix_quat(
                actions[:, EEAction.ee_left_quat_dims], 
                self.last_pub_action[EEAction.ee_left_quat_dims] if self.last_pub_action is not None else None
            )
            actions[:, EEAction.ee_right_quat_dims] = ActionEnsemble.fix_quat(
                actions[:, EEAction.ee_right_quat_dims], 
                self.last_pub_action[EEAction.ee_right_quat_dims] if self.last_pub_action is not None else None
            )
        
        self.action_buffer.append(EEAction(actions, actions_time, self.add_idx))
        self.add_idx += 1
        return int(np.argmax(mask))   # 返回有效action的第一帧action index

    def clear(self):
        self.action_buffer.clear()
        self.add_idx = 0
    @staticmethod
    def fix_quat(quat, base_quat=None):
        if base_quat is not None and np.sum(quat[0] * base_quat) < 0:
            quat[0] = -quat[0]
        dot = np.diag(np.dot(quat[0: -1], quat[1:].T))
        shift_indices = np.where(dot < 0)[0] + 1
        if len(shift_indices) % 2 == 1:
            shift_indices = np.append(shift_indices, len(quat))
        fixed_quat = quat.copy()
        for i in range(len(shift_indices) // 2):
            fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]] = -fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]]
        return fixed_quat

    def get_action(self, now):
        print(f"length: {len(self.action_buffer)}")
        while len(self.action_buffer) > 0:
            if not self.action_buffer[0].is_within(now):
                self.action_buffer.popleft()
            else:
                break
        idxs, steps, actions = [], [], []
        for i in range(len(self.action_buffer)):
            cur_idx, cur_step, cur_action = self.action_buffer[i].get_action(now)
            idxs.append(cur_idx)
            steps.append(cur_step)
            actions.append(cur_action)
        if len(actions) == 0:
            self.logger.info("Action buffer is empty")
            return

        actions = np.vstack(actions)
        if self.mode == "latest" or self.mode == "RTC":
            idxs = [idxs[-1]]
            steps = [steps[-1]]
            weights = np.array([1.0])
            esb_action = actions[-1]
        elif self.mode == "avg":
            weights = np.ones_like(actions)
            weights = weights / weights.sum()
            esb_action = self._weighted_average_action(actions, weights)
        elif self.mode == "ACT":
            weights = np.exp(self.k_act * np.arange(len(actions))) # NOTE: same as the old version of ensemble
            weights = weights / weights.sum()
            esb_action = self._weighted_average_action(actions, weights)
        elif self.mode == "HATO":
            weights = self.tau_hato ** np.arange(len(actions) - 1, -1, -1)
            weights = weights / weights.sum()
            esb_action = self._weighted_average_action(actions, weights)
        else:
            raise AttributeError("Ensemble mode {} not supported.".format(self.mode))
        delay = time.time() - now
        self.last_pub_action = esb_action
        self.logger.debug(f"Ensembled action idx: {self.get_idx} Action chunk num: {len(weights)} Idxs: {idxs} Steps: {steps} Weights: {np.around(weights, 2)}")
        self.get_idx += 1
        return esb_action

    def _weighted_average_action(self, actions, weights):
        """
        Weighted average action.
        """
        avg_action = np.zeros_like(actions[0])
        avg_action[EEAction.ee_left_trans_dims] = self._weighted_average_linear(actions[:, EEAction.ee_left_trans_dims], weights)
        avg_action[EEAction.ee_left_quat_dims] = self._weighted_average_quat(actions[:, EEAction.ee_left_quat_dims], weights)
        avg_action[EEAction.grp_left_dims] = self._weighted_average_linear(actions[:, EEAction.grp_left_dims], weights)
        avg_action[EEAction.ee_right_trans_dims] = self._weighted_average_linear(actions[:, EEAction.ee_right_trans_dims], weights)
        avg_action[EEAction.ee_right_quat_dims] = self._weighted_average_quat(actions[:, EEAction.ee_right_quat_dims], weights)
        avg_action[EEAction.grp_right_dims] = self._weighted_average_linear(actions[:, EEAction.grp_right_dims], weights)
        avg_action[EEAction.torso_dims] = self._weighted_average_linear(actions[:, EEAction.torso_dims], weights)
        return avg_action

    def _weighted_average_linear(self, linear, weights):
        return (linear * weights.reshape(-1, 1)).sum(0)

    def _weighted_average_quat(self, quats, weights):
        if self.use_slerp_quat_avg:
            avg_quat = quats[0]
            cumulative_weight = weights[0]
            for i in range(1, len(weights)):
                slerp = Slerp([0, 1], R.from_quat([avg_quat, quats[i]]))
                avg_quat = slerp(cumulative_weight / (cumulative_weight + weights[i])).as_quat()
                cumulative_weight += weights[i]
            return avg_quat
        else:
            avg_quat = np.sum(quats * weights[:, None], axis=0)
            avg_quat = avg_quat / np.linalg.norm(avg_quat)
            return avg_quat


        
        
