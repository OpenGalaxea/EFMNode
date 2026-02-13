import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from utils.message.datatype import Trajectory, ExecutionMode, RobotAction

class TrajectoryStitcher:
    def __init__(self, execution_mode: ExecutionMode, action_frequency: float = 15.0):
        self.execution_mode = execution_mode
        self.action_frequency = action_frequency

    def stitch(self, current_traj: Trajectory, new_traj: Trajectory) -> Trajectory:
        """
        将两个轨迹拼接在一起。
        找到 current_traj 和 new_traj 中最近的两个 action，然后：
        - 保留 current_traj 从开始到最近点的部分（包括最近点）
        - 从 new_traj 的最近点之后开始，添加后续的所有 actions（不包括最近点，避免重复）
        
        Args:
            current_traj: 当前轨迹
            new_traj: 新轨迹
            
        Returns:
            拼接后的新轨迹
        """
        # 如果任一轨迹为空，直接返回另一个轨迹的副本
        if len(current_traj.actions) == 0:
            stitched_traj = Trajectory()
            stitched_traj.timestamp = new_traj.timestamp
            # 直接引用 action，因为 ROS 消息对象通常不会被修改
            # 如果需要避免共享引用，可以使用 copy.deepcopy(action)
            for action in new_traj.actions:
                stitched_traj.actions.append(action)
            return stitched_traj
        
        if len(new_traj.actions) == 0:
            stitched_traj = Trajectory()
            stitched_traj.timestamp = current_traj.timestamp
            # 直接引用 action，因为 ROS 消息对象通常不会被修改
            # 如果需要避免共享引用，可以使用 copy.deepcopy(action)
            for action in current_traj.actions:
                stitched_traj.actions.append(action)
            return stitched_traj
        
        # 找到两个轨迹中最近的两个 action 索引
        current_idx, new_idx = self._find_nearest_action(current_traj, new_traj, self.action_frequency)
        
        # 创建新的轨迹对象
        stitched_traj = Trajectory()
        # 使用 current_traj 的时间戳作为拼接后轨迹的时间戳
        stitched_traj.timestamp = new_traj.timestamp
        
        # 保留 current_traj 从开始到最近点的部分（包括最近点）
        # 直接引用 action，因为 ROS 消息对象通常不会被修改
        # 如果需要避免共享引用，可以使用 copy.deepcopy(current_traj.actions[i])
        # deque 的 maxlen 会自动处理长度限制
        for i in range(current_idx + 1):
            if i > len(current_traj.actions) - 1:
                break
            stitched_traj.actions.append(current_traj.actions[i])
        
        # 从 new_traj 的最近点之后开始，添加后续的所有 actions（不包括最近点，避免重复）
        # 直接引用 action，因为 ROS 消息对象通常不会被修改
        # 如果需要避免共享引用，可以使用 copy.deepcopy(new_traj.actions[i])
        # deque 的 maxlen 会自动处理长度限制
        for i in range(new_idx + 1, len(new_traj.actions)):
            stitched_traj.actions.append(new_traj.actions[i])
        
        return stitched_traj

    def _find_nearest_action(self, current_traj: Trajectory, new_traj: Trajectory, action_frequency: float = 15.0) -> Tuple[int, int]:
        """
        找到两个轨迹中最近的两个 action。
        
        Args:
            current_traj: 当前轨迹
            new_traj: 新轨迹
            action_frequency: action 的频率，用于计算速度和加速度，默认 15Hz
        
        Returns:
            (current_idx, new_idx): current_traj 中最近点的索引和 new_traj 中最近点的索引
        """
        if self.execution_mode == ExecutionMode.EE_POSE:
            return self._find_nearest_action_ee_pose(current_traj, new_traj, action_frequency)
        elif self.execution_mode == ExecutionMode.JOINT_STATE:
            return self._find_nearest_action_joint_state(current_traj, new_traj, action_frequency)
        else:
            raise ValueError(f"Invalid execution mode: {self.execution_mode}")

    def _find_nearest_action_ee_pose(self, current_traj: Trajectory, new_traj: Trajectory, action_frequency: float = 15.0) -> Tuple[int, int]:
        """
        使用 KD 树找到两个轨迹中最近的两个 RobotAction（基于 EE_POSE）。
        距离计算基于位置和方向（四元数），包括位置、速度、加速度和角速度、角加速度。
        使用 scipy.spatial.transform.Rotation 处理四元数转换。
        
        Args:
            current_traj: 当前轨迹
            new_traj: 新轨迹
            action_frequency: action 的频率（Hz），用于计算速度和加速度，默认 15Hz
            
        Returns:
            (current_idx, new_idx): current_traj 中最近点的索引和 new_traj 中最近点的索引
        """
        # 定义 EE_POSE 字段的顺序
        ee_pose_fields = ['left_ee_pose', 'right_ee_pose']
        
        # 每个 EE pose 的维度：位置(3) + 位置速度(3) + 位置加速度(3) + 旋转向量(3) + 角速度(3) + 角加速度(3) = 18
        dim_per_ee_pose = 18
        total_dim = len(ee_pose_fields) * dim_per_ee_pose  # 36 维（两个 EE pose）
        
        # 计算时间步长
        dt = 1.0 / action_frequency if action_frequency > 0 else 1.0 / 15.0
        
        def compute_velocity_acceleration(sequence: list, dt: float) -> Tuple[list, list]:
            """
            根据序列计算速度和加速度（适用于位置或旋转向量）。
            
            Args:
                sequence: 序列，每个元素是一个 numpy 数组
                dt: 时间步长
                
            Returns:
                (velocities, accelerations): 速度和加速度数组列表
            """
            n = len(sequence)
            if n == 0:
                return [], []
            
            velocities = []
            accelerations = []
            
            for i in range(n):
                if n == 1:
                    # 只有一个点，速度和加速度都为 0
                    vel = np.zeros_like(sequence[0])
                    acc = np.zeros_like(sequence[0])
                elif i == 0:
                    # 第一个点：前向差分计算速度
                    vel = (sequence[1] - sequence[0]) / dt
                    # 加速度使用二阶前向差分
                    if n >= 3:
                        acc = (sequence[2] - 2 * sequence[1] + sequence[0]) / (dt * dt)
                    else:
                        acc = np.zeros_like(sequence[0])
                elif i == n - 1:
                    # 最后一个点：后向差分计算速度
                    vel = (sequence[i] - sequence[i-1]) / dt
                    # 加速度使用二阶后向差分
                    if n >= 3:
                        acc = (sequence[i] - 2 * sequence[i-1] + sequence[i-2]) / (dt * dt)
                    else:
                        acc = np.zeros_like(sequence[i])
                else:
                    # 中间点：中心差分计算速度
                    vel = (sequence[i+1] - sequence[i-1]) / (2 * dt)
                    # 加速度使用二阶中心差分
                    acc = (sequence[i+1] - 2 * sequence[i] + sequence[i-1]) / (dt * dt)
                
                velocities.append(vel)
                accelerations.append(acc)
            
            return velocities, accelerations
        
        def extract_ee_pose_vector(action: RobotAction, action_idx: int, 
                                   trajectory_actions: list) -> np.ndarray:
            """
            从 RobotAction 中提取所有 EE_POSE 的位置、速度、加速度和旋转信息。
            
            Args:
                action: 当前的 RobotAction
                action_idx: 当前 action 在轨迹中的索引
                trajectory_actions: 整个轨迹的 actions 列表
            """
            vector_parts = []
            
            for field_name in ee_pose_fields:
                # 提取该字段在所有 action 中的位姿序列
                pose_sequence = []
                rotation_vector_sequence = []
                
                for traj_action in trajectory_actions:
                    pose_stamped = getattr(traj_action, field_name, None)
                    if pose_stamped is not None and hasattr(pose_stamped, 'pose'):
                        # 提取位置 [x, y, z]
                        pos = np.array([
                            pose_stamped.pose.position.x,
                            pose_stamped.pose.position.y,
                            pose_stamped.pose.position.z
                        ], dtype=np.float64)
                        pose_sequence.append(pos)
                        
                        # 提取四元数 [qx, qy, qz, qw] 并转换为旋转向量
                        quat = np.array([
                            pose_stamped.pose.orientation.x,
                            pose_stamped.pose.orientation.y,
                            pose_stamped.pose.orientation.z,
                            pose_stamped.pose.orientation.w
                        ], dtype=np.float64)
                        
                        # 使用 scipy.spatial.transform.Rotation 将四元数转换为旋转向量
                        # Rotation 期望四元数格式为 [x, y, z, w]
                        try:
                            rot = Rotation.from_quat(quat)
                            # 转换为旋转向量（axis-angle 表示，3D）
                            rot_vec = rot.as_rotvec()
                            rotation_vector_sequence.append(rot_vec)
                        except Exception:
                            # 如果转换失败（例如四元数无效），使用零向量
                            rotation_vector_sequence.append(np.zeros(3, dtype=np.float64))
                    else:
                        # 如果字段为 None，用 0 填充
                        pose_sequence.append(np.zeros(3, dtype=np.float64))
                        rotation_vector_sequence.append(np.zeros(3, dtype=np.float64))
                
                # 计算位置的速度和加速度
                if len(pose_sequence) > 0:
                    pos_velocities, pos_accelerations = compute_velocity_acceleration(pose_sequence, dt)
                    current_pos = pose_sequence[action_idx] if action_idx < len(pose_sequence) else np.zeros(3, dtype=np.float64)
                    current_pos_vel = pos_velocities[action_idx] if action_idx < len(pos_velocities) else np.zeros(3, dtype=np.float64)
                    current_pos_acc = pos_accelerations[action_idx] if action_idx < len(pos_accelerations) else np.zeros(3, dtype=np.float64)
                else:
                    current_pos = np.zeros(3, dtype=np.float64)
                    current_pos_vel = np.zeros(3, dtype=np.float64)
                    current_pos_acc = np.zeros(3, dtype=np.float64)
                
                # 计算旋转向量的角速度和角加速度
                if len(rotation_vector_sequence) > 0:
                    ang_velocities, ang_accelerations = compute_velocity_acceleration(rotation_vector_sequence, dt)
                    current_rot_vec = rotation_vector_sequence[action_idx] if action_idx < len(rotation_vector_sequence) else np.zeros(3, dtype=np.float64)
                    current_ang_vel = ang_velocities[action_idx] if action_idx < len(ang_velocities) else np.zeros(3, dtype=np.float64)
                    current_ang_acc = ang_accelerations[action_idx] if action_idx < len(ang_accelerations) else np.zeros(3, dtype=np.float64)
                else:
                    current_rot_vec = np.zeros(3, dtype=np.float64)
                    current_ang_vel = np.zeros(3, dtype=np.float64)
                    current_ang_acc = np.zeros(3, dtype=np.float64)
                
                # 按顺序拼接：位置、位置速度、位置加速度、旋转向量、角速度、角加速度
                vector_parts.extend([
                    current_pos,           # 3D
                    current_pos_vel,       # 3D
                    current_pos_acc,       # 3D
                    current_rot_vec,       # 3D
                    current_ang_vel,       # 3D
                    current_ang_acc        # 3D
                ])
            
            if len(vector_parts) == 0:
                return np.zeros(total_dim, dtype=np.float64)
            return np.concatenate(vector_parts)
        
        # 提取 current_traj 中所有 action 的特征向量
        current_actions_list = list(current_traj.actions)
        current_vectors = []
        current_indices = []
        for idx, action in enumerate(current_actions_list):
            vector = extract_ee_pose_vector(action, idx, current_actions_list)
            current_vectors.append(vector)
            current_indices.append(idx)
        
        if len(current_vectors) == 0:
            return (0, 0)
        
        # 提取 new_traj 中所有 action 的特征向量
        new_actions_list = list(new_traj.actions)
        new_vectors = []
        new_indices = []
        for idx, action in enumerate(new_actions_list):
            vector = extract_ee_pose_vector(action, idx, new_actions_list)
            new_vectors.append(vector)
            new_indices.append(idx)
        
        if len(new_vectors) == 0:
            return (0, 0)
        
        # 构建 KD 树（使用 new_traj 的向量作为数据点）
        new_vectors_array = np.array(new_vectors)
        kdtree = cKDTree(new_vectors_array)
        
        # 对于 current_traj 中的每个向量，找到 new_traj 中最近的向量
        min_distances = []
        min_indices = []
        
        for current_vec in current_vectors:
            # 查询最近邻（k=1）
            distance, index = kdtree.query(current_vec, k=1)
            min_distances.append(distance)
            min_indices.append(index)
        
        # 找到全局最小距离对应的索引对
        global_min_idx = np.argmin(min_distances)
        current_nearest_idx = current_indices[global_min_idx]  # current_traj 中的索引
        nearest_new_idx = min_indices[global_min_idx]  # new_traj 中的索引（在 new_vectors 中的索引）
        
        # 返回两个轨迹中对应的原始索引
        return (current_nearest_idx, new_indices[nearest_new_idx])

    def _find_nearest_action_joint_state(self, current_traj: Trajectory, new_traj: Trajectory, action_frequency: float = 15.0) -> Tuple[int, int]:
        """
        使用 KD 树找到两个轨迹中最近的两个 RobotAction。
        距离计算基于所有非 None 的 JointState 类型的数据，包括位置、速度和加速度。
        速度和加速度根据位置序列和频率计算得出。
        
        Args:
            current_traj: 当前轨迹
            new_traj: 新轨迹
            action_frequency: action 的频率（Hz），用于计算速度和加速度，默认 15Hz
            
        Returns:
            (current_idx, new_idx): current_traj 中最近点的索引和 new_traj 中最近点的索引
        """
        # 定义 JointState 字段的顺序（用于构建特征向量）
        joint_state_fields = ['left_arm', 'right_arm', 'torso', 
                             'left_gripper', 'right_gripper', 'chassis']
        
        # 计算时间步长
        dt = 1.0 / action_frequency if action_frequency > 0 else 1.0 / 15.0
        
        # 首先遍历所有 action 来确定每个字段的最大维度
        # 这样可以确保所有向量具有相同的维度结构
        field_dims = {}
        for action in list(current_traj.actions) + list(new_traj.actions):
            for field_name in joint_state_fields:
                joint_state = getattr(action, field_name, None)
                if joint_state is not None and hasattr(joint_state, 'position'):
                    dim = len(joint_state.position)
                    if field_name not in field_dims:
                        field_dims[field_name] = dim
                    else:
                        field_dims[field_name] = max(field_dims[field_name], dim)
        
        # 计算总维度（位置 + 速度 + 加速度，每个都是 3 倍）
        total_dim = sum(field_dims.values()) * 3 if field_dims else 0
        if total_dim == 0:
            return (0, 0)
        
        def compute_velocity_acceleration(position_sequence: list, dt: float) -> Tuple[np.ndarray, np.ndarray]:
            """
            根据位置序列计算速度和加速度。
            
            Args:
                position_sequence: 位置序列，每个元素是一个 numpy 数组
                dt: 时间步长
                
            Returns:
                (velocities, accelerations): 速度和加速度数组列表
            """
            n = len(position_sequence)
            if n == 0:
                return [], []
            
            velocities = []
            accelerations = []
            
            for i in range(n):
                if n == 1:
                    # 只有一个点，速度和加速度都为 0
                    vel = np.zeros_like(position_sequence[0])
                    acc = np.zeros_like(position_sequence[0])
                elif i == 0:
                    # 第一个点：前向差分计算速度
                    vel = (position_sequence[1] - position_sequence[0]) / dt
                    # 加速度使用二阶前向差分
                    if n >= 3:
                        acc = (position_sequence[2] - 2 * position_sequence[1] + position_sequence[0]) / (dt * dt)
                    else:
                        acc = np.zeros_like(position_sequence[0])
                elif i == n - 1:
                    # 最后一个点：后向差分计算速度
                    vel = (position_sequence[i] - position_sequence[i-1]) / dt
                    # 加速度使用二阶后向差分
                    if n >= 3:
                        acc = (position_sequence[i] - 2 * position_sequence[i-1] + position_sequence[i-2]) / (dt * dt)
                    else:
                        acc = np.zeros_like(position_sequence[i])
                else:
                    # 中间点：中心差分计算速度
                    vel = (position_sequence[i+1] - position_sequence[i-1]) / (2 * dt)
                    # 加速度使用二阶中心差分
                    acc = (position_sequence[i+1] - 2 * position_sequence[i] + position_sequence[i-1]) / (dt * dt)
                
                velocities.append(vel)
                accelerations.append(acc)
            
            return velocities, accelerations
        
        def extract_joint_state_vector(action: RobotAction, action_idx: int, 
                                      trajectory_actions: list, field_dims: dict) -> np.ndarray:
            """
            从 RobotAction 中提取所有 JointState 的位置、速度和加速度数据，按固定顺序拼接成特征向量。
            如果某个字段为 None，则用 0 填充该字段对应的维度。
            
            Args:
                action: 当前的 RobotAction
                action_idx: 当前 action 在轨迹中的索引
                trajectory_actions: 整个轨迹的 actions 列表
                field_dims: 每个字段的维度字典
            """
            vector_parts = []
            
            for field_name in joint_state_fields:
                # 提取该字段在所有 action 中的位置序列
                position_sequence = []
                for traj_action in trajectory_actions:
                    joint_state = getattr(traj_action, field_name, None)
                    if joint_state is not None and hasattr(joint_state, 'position'):
                        pos_array = np.array(joint_state.position, dtype=np.float64)
                        expected_dim = field_dims.get(field_name, len(pos_array))
                        # 调整维度
                        if len(pos_array) < expected_dim:
                            pos_array = np.pad(pos_array, (0, expected_dim - len(pos_array)), 
                                             mode='constant', constant_values=0.0)
                        elif len(pos_array) > expected_dim:
                            pos_array = pos_array[:expected_dim]
                        position_sequence.append(pos_array)
                    else:
                        # 如果字段为 None，用 0 填充
                        expected_dim = field_dims.get(field_name, 0)
                        if expected_dim > 0:
                            position_sequence.append(np.zeros(expected_dim, dtype=np.float64))
                
                # 计算速度和加速度
                if len(position_sequence) > 0:
                    velocities, accelerations = compute_velocity_acceleration(position_sequence, dt)
                    
                    # 获取当前 action 的位置、速度和加速度
                    current_pos = position_sequence[action_idx] if action_idx < len(position_sequence) else np.zeros(field_dims.get(field_name, 0), dtype=np.float64)
                    current_vel = velocities[action_idx] if action_idx < len(velocities) else np.zeros(field_dims.get(field_name, 0), dtype=np.float64)
                    current_acc = accelerations[action_idx] if action_idx < len(accelerations) else np.zeros(field_dims.get(field_name, 0), dtype=np.float64)
                else:
                    expected_dim = field_dims.get(field_name, 0)
                    current_pos = np.zeros(expected_dim, dtype=np.float64)
                    current_vel = np.zeros(expected_dim, dtype=np.float64)
                    current_acc = np.zeros(expected_dim, dtype=np.float64)
                
                # 拼接位置、速度、加速度
                vector_parts.append(current_pos)
                vector_parts.append(current_vel)
                vector_parts.append(current_acc)
            
            if len(vector_parts) == 0:
                return np.zeros(total_dim, dtype=np.float64)
            return np.concatenate(vector_parts)
        
        # 提取 current_traj 中所有 action 的特征向量
        current_actions_list = list(current_traj.actions)
        current_vectors = []
        current_indices = []
        for idx, action in enumerate(current_actions_list):
            vector = extract_joint_state_vector(action, idx, current_actions_list, field_dims)
            current_vectors.append(vector)
            current_indices.append(idx)
        
        if len(current_vectors) == 0:
            return (0, 0)
        
        # 提取 new_traj 中所有 action 的特征向量
        new_actions_list = list(new_traj.actions)
        new_vectors = []
        new_indices = []
        for idx, action in enumerate(new_actions_list):
            vector = extract_joint_state_vector(action, idx, new_actions_list, field_dims)
            new_vectors.append(vector)
            new_indices.append(idx)
        
        if len(new_vectors) == 0:
            return (0, 0)
        
        # 构建 KD 树（使用 new_traj 的向量作为数据点）
        new_vectors_array = np.array(new_vectors)
        kdtree = cKDTree(new_vectors_array)
        
        # 对于 current_traj 中的每个向量，找到 new_traj 中最近的向量
        min_distances = []
        min_indices = []
        
        for current_vec in current_vectors:
            # 查询最近邻（k=1）
            distance, index = kdtree.query(current_vec, k=1)
            min_distances.append(distance)
            min_indices.append(index)
        
        # 找到全局最小距离对应的索引对
        global_min_idx = np.argmin(min_distances)
        current_nearest_idx = current_indices[global_min_idx]  # current_traj 中的索引
        nearest_new_idx = min_indices[global_min_idx]  # new_traj 中的索引（在 new_vectors 中的索引）
        
        # 返回两个轨迹中对应的原始索引
        return (current_nearest_idx, new_indices[nearest_new_idx])
