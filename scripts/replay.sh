bag=$1
ros2 bag play $bag --remap \
    /motion_target/target_pose_arm_left:=/motion_target/target_pose_arm_left_gt \
    /motion_target/target_pose_arm_right:=/motion_target/target_pose_arm_right_gt \
    /motion_target/target_joint_state_arm_left:=/motion_target/target_joint_state_arm_left_gt \
    /motion_target/target_joint_state_arm_right:=/motion_target/target_joint_state_arm_right_gt \
    /motion_target/target_position_gripper_left:=/motion_target/target_position_gripper_left_gt \
    /motion_target/target_position_gripper_right:=/motion_target/target_position_gripper_right_gt \
    /motion_target/target_joint_state_torso:=/motion_target/target_joint_state_torso_gt \
    -s mcap