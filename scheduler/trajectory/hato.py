import numpy as np
from utils.message.datatype import EEAction, JointAction
from scipy.spatial.transform import Rotation as R, Slerp
from utils.message.datatype import EEAction, JointAction, ExecutionMode

def weighted_average_linear(linear, weights):
    return (linear * weights.reshape(-1, 1)).sum(0)
def weighted_average_quat(quats, weights):
    avg_quat = quats[0]
    cumulative_weight = weights[0]
    for i in range(1, len(weights)):
        slerp = Slerp([0, 1], R.from_quat([avg_quat, quats[i]]))
        avg_quat = slerp(cumulative_weight / (cumulative_weight + weights[i])).as_quat()
        cumulative_weight += weights[i]
    return avg_quat

def weighted_average_ee_action(actions, weights):
    """
    Weighted average action.
    """
    avg_action = np.zeros_like(actions[0])
    avg_action[EEAction.ee_left_trans_dims] = weighted_average_linear(actions[:, EEAction.ee_left_trans_dims], weights)
    avg_action[EEAction.ee_left_quat_dims] = weighted_average_quat(actions[:, EEAction.ee_left_quat_dims], weights)
    avg_action[EEAction.grp_left_dims] = weighted_average_linear(actions[:, EEAction.grp_left_dims], weights)
    avg_action[EEAction.ee_right_trans_dims] = weighted_average_linear(actions[:, EEAction.ee_right_trans_dims], weights)
    avg_action[EEAction.ee_right_quat_dims] = weighted_average_quat(actions[:, EEAction.ee_right_quat_dims], weights)
    avg_action[EEAction.grp_right_dims] = weighted_average_linear(actions[:, EEAction.grp_right_dims], weights)
    avg_action[EEAction.torso_dims] = weighted_average_linear(actions[:, EEAction.torso_dims], weights)
    return avg_action

def weighted_average_joint_action(actions, weights):
    """
    Weighted average action.
    """
    avg_action = np.zeros_like(actions[0])
    avg_action[JointAction.left_arm_joint_dims] = weighted_average_linear(actions[:, JointAction.left_arm_joint_dims], weights)
    avg_action[JointAction.grp_left_dims] = weighted_average_linear(actions[:, JointAction.grp_left_dims], weights)
    avg_action[JointAction.right_arm_joint_dims] = weighted_average_linear(actions[:, JointAction.right_arm_joint_dims], weights)
    avg_action[JointAction.grp_right_dims] = weighted_average_linear(actions[:, JointAction.grp_right_dims], weights)
    return avg_action
        
def ensemble(actions, execution_mode, tau_hato=0.5):
    actions = np.vstack(actions)
    weights = tau_hato ** np.arange(len(actions) - 1, -1, -1)
    weights = weights / weights.sum()
    if execution_mode == ExecutionMode.EE_POSE:
        esb_action = weighted_average_ee_action(actions, weights)
    elif execution_mode == ExecutionMode.JOINT_STATE:
        esb_action = weighted_average_joint_action(actions, weights)
    else:
        raise NotImplementedError

    return esb_action
