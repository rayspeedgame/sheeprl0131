import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sheeprl.envs.envfunc.UAV import get_observed_density, allocate_uav_service
from sheeprl.envs.envfunc.Crowd import get_crowd_density, get_positions

class UAVEnvWrapper(gym.Env):
    def __init__(self, config_path="config.json", db_path="exhibition2.sqlite"):
        super(UAVEnvWrapper, self).__init__()
        
        # 从配置文件加载参数
        with open(config_path) as f:
            self.config = json.load(f)
        
        # 数据库路径
        self.db_path = db_path
        
        # 环境参数
        self.n_uav = self.config["n_uav"]
        self.max_steps = self.config["max_steps"]
        self.fps = self.config["fps"]
        self.boundary = np.array(self.config["boundary"])
        self.density_size = self.config["density_matrix_size"]
        
        # 奖励参数
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.gamma = self.config["gamma"]
        
        # 定义动作空间 (n_uav * 3 的一维连续值)
        self.action_space = spaces.Box(
            low=np.array([self.config["min_action"]] * (3 * self.n_uav)),
            high=np.array([self.config["max_action"]] * (3 * self.n_uav)),
            dtype=np.float32
        )
        
        # 更新观测空间 - 仅包含人群密度信息
        self.observation_space = spaces.Dict({
            "density_matrix": spaces.Box(
                low=0,
                high=self.config["max_density"],
                shape=(1, self.density_size, self.density_size),  # 添加通道维度
                dtype=np.float32
            )
        })
        
        # 初始化状态
        self.current_step = 0
        self.uav_positions = None
        # 创建包含id的无人机状态矩阵
        self.uav_states = None
        
        # 添加缓存变量，避免重复计算
        self.frame_id = None
        self.full_density = None
        self.observed_density = None
        self.observation_mask = None
        self.observed_ratio = None
        self.observed_area_ratio = None
        self.total_people = None
        self.observed_people = None
        self.reward = None
        self.total_power = None  # 添加总发射功率
        self.served_people = None  # 添加服务人数
        self.service_ratio = None  # 添加服务比例

    def _get_info(self):
        """返回当前环境的额外信息
        
        Returns:
            dict: 包含以下信息:
                - uav_positions: 所有无人机的当前位置 (n_uav * 3)
                - full_density: 完整的人群密度矩阵 (1 x density_size x density_size) 
                - frame_id: 当前帧ID
                - observed_ratio: 观测到的人群占总人群的比例
                - observed_area_ratio: 无人机观测范围占整个场景的比例
                - total_power: 无人机总发射功率
                - served_people: 服务的人数
                - service_ratio: 服务人群占总人群的比例
        """
        # 将无人机位置展平为一维数组
        flat_uav_positions = self.uav_positions.flatten()
        
        # 将完整密度矩阵添加通道维度（如果尚未添加）
        if self.full_density.ndim == 2:
            full_density_with_channel = self.full_density.astype(np.float32)[np.newaxis, :, :]
        else:
            full_density_with_channel = self.full_density
        
        return {
            # 确保返回的是numpy数组类型
            "uav_positions": np.array(flat_uav_positions, dtype=np.float32),
            "full_density": full_density_with_channel,
            "frame_id": self.frame_id,
            "observed_ratio": np.float32(self.observed_ratio),
            "observed_area_ratio": np.float32(self.observed_area_ratio),
            "total_power": np.float32(self.total_power if self.total_power is not None else 0.0),
            "served_people": np.float32(self.served_people if self.served_people is not None else 0.0),
            "service_ratio": np.float32(self.service_ratio if self.service_ratio is not None else 0.0)
        }

    def _update_state(self):
        """更新环境状态和相关计算值"""
        # 计算帧ID
        self.frame_id = int(self.current_step * self.fps)
        
        # 获取完整的密度分布
        self.full_density = get_crowd_density(
            self.density_size,
            self.density_size,
            self.frame_id,
            db_path=self.db_path
        )
        
        # 获取观测到的密度矩阵和观测掩码
        self.observed_density, self.observation_mask = get_observed_density(
            self.frame_id, 
            self.uav_states, 
            self.density_size, 
            self.density_size,
            db_path=self.db_path
        )
        
        # 计算各种统计数据
        self.total_people = np.sum(self.full_density)
        self.observed_people = np.sum(self.observed_density)
        self.observed_ratio = self.observed_people / self.total_people if self.total_people > 0 else 0
        
        # 计算观测区域比例
        observed_area = np.count_nonzero(self.observation_mask)
        total_area = self.density_size * self.density_size
        self.observed_area_ratio = observed_area / total_area

    def reset(self, seed=None, options=None):
        # 重置环境时间
        self.current_step = 10  # 根据需求初始化为10
        
        # 随机初始化无人机位置
        self.uav_positions = np.random.uniform(
            low=self.boundary[0],
            high=self.boundary[1],
            size=(self.n_uav, 3)
        )
        
        # 初始化包含id的无人机状态矩阵
        self.uav_states = np.zeros((self.n_uav, 4))
        self.uav_states[:, 0] = np.arange(self.n_uav)  # 设置id
        self.uav_states[:, 1:4] = self.uav_positions   # 设置位置
        
        # 更新环境状态和缓存值
        self._update_state()
        
        # 初始化能量和服务人数
        self.total_power, self.served_people = allocate_uav_service(
            self.frame_id, 
            self.uav_states,
            db_path=self.db_path
        )
        
        # 计算服务用户比例
        self.service_ratio = self.served_people / self.total_people if self.total_people > 0 else 0
        
        return self._get_obs(), self._get_info()

    def step(self, actions):
        self.current_step += 1
        
        # 1. 将一维动作重塑为(n_uav, 3)形状
        actions_reshaped = actions.reshape(self.n_uav, 3)
        
        # 更新无人机位置
        self.uav_positions = np.clip(
            self.uav_positions + actions_reshaped,
            self.boundary[0],
            self.boundary[1]
        )
        
        # 更新无人机状态矩阵
        self.uav_states[:, 1:4] = self.uav_positions
        
        # 更新环境状态和缓存的计算值
        self._update_state()
        
        # 计算服务人数和能量消耗
        self.total_power, self.served_people = allocate_uav_service(
            self.frame_id, 
            self.uav_states,
            db_path=self.db_path
        )
        
        # 计算服务用户比例
        self.service_ratio = self.served_people / self.total_people if self.total_people > 0 else 0
        
        # 奖励计算
        decay_factor = np.exp(-0.01 * self.current_step)  # 指数衰减因子
        epsilon = 1e-6  # 防止除零
        
        # 观测奖励保持不变
        observation_reward = self.alpha * self.observed_ratio * decay_factor
        
        # 服务用户奖励基于服务用户比例
        service_reward = self.beta * self.service_ratio
        
        # 功率惩罚根据服务率动态调整权重
        power_penalty_weight = self.gamma
        if self.service_ratio > 0.9:  # 当服务率超过90%时
            # 随着服务率提高，增加功率惩罚的权重（最大可达到原权重的2倍）
            power_penalty_weight = self.gamma * (1 + (self.service_ratio - 0.9) * 10)
        
        # 功率惩罚与服务率相关
        power_penalty = power_penalty_weight * (self.total_power / ((self.service_ratio**2) * self.total_people + epsilon))
        
        self.reward = observation_reward + service_reward - power_penalty
        
        # 检查终止条件
        terminated = self.current_step >= self.max_steps
        truncated = False  # 可根据需要添加其他终止条件
        
        return self._get_obs(), self.reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        # 将密度矩阵增加一个通道维度 [height, width] -> [1, height, width]
        observed_density_with_channel = self.observed_density.astype(np.float32)[np.newaxis, :, :]
        
        # 仅返回人群密度信息（带通道维度）
        return {
            "density_matrix": observed_density_with_channel
        }

    def get_uav_positions(self):
        # 返回无人机位置（一维形式）
        return self.uav_positions.flatten().copy()

    def render(self, mode='human'):
        # 可添加可视化逻辑
        pass
