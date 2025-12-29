bag=$1

# 检查目标路径是否存在，如果存在就删除
if [ -d "$bag" ]; then
    echo "目标路径 '$bag' 已存在，正在删除..."
    rm -rf "$bag"
    echo "已删除目标路径 '$bag'"
fi

ros2 bag record -s mcap \
    /hdas/camera_head/depth/depth_registered \
    /hdas/camera_head/left_raw/image_raw_color/compressed \
    /hdas/camera_wrist_left/color/image_rect_raw/compressed \
    /hdas/camera_wrist_right/color/image_rect_raw/compressed \
    /hdas/feedback_gripper_left \
    /hdas/feedback_gripper_right \
    /hdas/feedback_torso \
    /hdas/feedback_arm_left \
    /hdas/feedback_arm_right \
    /hdas/feedback_status_arm_left \
    /hdas/feedback_status_arm_right \
    /hdas/feedback_status_torso \
    /hdas/feedback_chassis \
    /motion_control/control_arm_left \
    /motion_control/control_arm_right \
    /motion_control/control_gripper_left \
    /motion_control/control_gripper_right \
    /motion_control/control_chassis \
    /motion_control/control_torso \
    /motion_control/pose_ee_arm_left \
    /motion_control/pose_ee_arm_right \
    /motion_control/pose_floating_base \
    /motion_target/target_position_gripper_left \
    /motion_target/target_position_gripper_right \
    /motion_target/brake_mode \
    /motion_target/chassis_acc_limit \
    /motion_target/target_joint_state_arm_left \
    /motion_target/target_joint_state_arm_right \
    /motion_target/target_joint_state_torso \
    /motion_target/target_pose_arm_left \
    /motion_target/target_pose_arm_right \
    /motion_target/target_pose_arm_left_sdk \
    /motion_target/target_pose_arm_right_sdk \
    /motion_target/target_pose_arm_left_raw \
    /motion_target/target_pose_arm_right_raw \
    /motion_target/target_position_gripper_left_sdk \
    /motion_target/target_position_gripper_right_sdk \
    /motion_target/target_position_gripper_left_raw \
    /motion_target/target_position_gripper_right_raw \
    /motion_target/target_pose_arm_left_gt \
    /motion_target/target_pose_arm_right_gt \
    /motion_target/target_position_gripper_left_gt \
    /motion_target/target_position_gripper_right_gt \
    /motion_target/target_speed_chassis \
    /relaxed_ik/motion_control/pose_ee_arm_left \
    /relaxed_ik/motion_control/pose_ee_arm_right \
    /motion_target/target_joint_state_torso_gt \
    /motion_target/target_joint_state_torso_raw \
    /progress \
    /tf \
    --output $bag