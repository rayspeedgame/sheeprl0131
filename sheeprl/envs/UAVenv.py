import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sheeprl.envs.envfunc.UAV import get_observed_density, allocate_uav_service
from sheeprl.envs.envfunc.Crowd import get_crowd_density

class UAVEnvWrapper(gym.Env):
    def __init__(self, config_path="config.json"):
        super(UAVEnvWrapper, self).__init__()
        
        # 从配置文件加载参数
        with open(config_path) as f:
            self.config = json.load(f)
        
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
        
        # 定义动作空间 (n_uav x 3 的连续值)
        self.action_space = spaces.Box(
            low=np.array([self.config["min_action"]]*3*self.n_uav).reshape(self.n_uav,3),
            high=np.array([self.config["max_action"]]*3*self.n_uav).reshape(self.n_uav,3),
            dtype=np.float32
        )
        
        # 更新观测空间 - 仅包含人群密度信息
        self.observation_space = spaces.Dict({
            "density_matrix": spaces.Box(
                low=0,
                high=self.config["max_density"],
                shape=(self.density_size, self.density_size),
                dtype=np.int32
            )
        })
        
        # 初始化状态
        self.current_step = 0
        self.uav_positions = None
        # 创建包含id的无人机状态矩阵
        self.uav_states = None

    def _get_info(self):
        """返回当前环境的额外信息
        
        Returns:
            dict: 包含以下信息:
                - uav_positions: 所有无人机的当前位置 (n_uav x 3)
                - full_density: 完整的人群密度矩阵 (density_size x density_size) 
                - frame_id: 当前帧ID
                - observed_ratio: 观测到的人群占总人群的比例
                - observed_area_ratio: 无人机观测范围占整个场景的比例
        """
        # 计算当前帧ID
        frame_id = int(self.current_step * self.fps)
        
        # 获取完整的密度分布
        full_density = get_crowd_density(
            self.density_size,
            self.density_size,
            frame_id
        )
        
        # 获取观测到的密度矩阵和观测掩码
        observed_density, observation_mask = get_observed_density(
            frame_id, 
            self.uav_states, 
            self.density_size, 
            self.density_size
        )
        
        # 计算观测到的人群比例
        total_people = np.sum(full_density)
        observed_people = np.sum(observed_density)
        observed_ratio = observed_people / total_people if total_people > 0 else 0
        
        # 计算观测区域比例 (使用观测掩码)
        observed_area = np.count_nonzero(observation_mask)
        total_area = self.density_size * self.density_size
        observed_area_ratio = observed_area / total_area
        
        return {
            "uav_positions": self.uav_positions.copy(),
            "full_density": full_density,
            "frame_id": frame_id,
            "observed_ratio": observed_ratio,
            "observed_area_ratio": observed_area_ratio
        }

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
        
        return self._get_obs(), self._get_info()

    def step(self, actions):
        self.current_step += 1
        
        # 1. 更新无人机位置
        self.uav_positions = np.clip(
            self.uav_positions + actions,
            self.boundary[0],
            self.boundary[1]
        )
        
        # 更新无人机状态矩阵
        self.uav_states[:, 1:4] = self.uav_positions
        
        # 2. 计算帧ID
        frame_id = int(self.current_step * self.fps)
        
        # 3. 计算奖励 - 直接使用更新后的无人机位置
        total_power, served_people = allocate_uav_service(
            frame_id, 
            self.uav_states
        )
        
        # 获取完整的密度分布
        full_density = get_crowd_density(
            self.density_size, 
            self.density_size, 
            frame_id
        )
        total_people = np.sum(full_density)
        
        # 计算观测覆盖区域中的人数比例
        observed_density, _ = get_observed_density(
            frame_id, 
            self.uav_states, 
            self.density_size, 
            self.density_size
        )
        observed_people = np.sum(observed_density)
        observed_ratio = observed_people / total_people if total_people > 0 else 0
        
        # 奖励计算
        decay_factor = np.exp(-0.01 * self.current_step)  # 指数衰减因子
        epsilon = 1e-6  # 防止除零
        
        reward = (self.alpha * observed_ratio * decay_factor) + \
                 (self.beta * served_people) - \
                 (self.gamma * (total_power / (served_people**2 + epsilon)))
        
        # 4. 检查终止条件
        terminated = self.current_step >= self.max_steps
        truncated = False  # 可根据需要添加其他终止条件
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        # 计算帧ID
        frame_id = int(self.current_step * self.fps)
        
        # 获取观测到的密度矩阵
        observed_density, _ = get_observed_density(
            frame_id, 
            self.uav_states, 
            self.density_size, 
            self.density_size
        )
        
        # 仅返回人群密度信息
        return {
            "density_matrix": observed_density
        }

    def get_uav_positions(self):
        # 返回无人机位置
        return self.uav_positions.copy()

    def render(self, mode='human'):
        # 可添加可视化逻辑
        pass
