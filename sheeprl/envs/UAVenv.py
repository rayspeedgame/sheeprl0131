import json
import numpy as np
import os
import glob
import gymnasium as gym
import sqlite3
from gymnasium import spaces
from sheeprl.envs.envfunc.UAV import get_observed_density, allocate_uav_service
from sheeprl.envs.envfunc.Crowd import get_crowd_density, get_positions

class UAVEnvWrapper(gym.Env):
    def __init__(self, config_path="config.json", db_path="crowd_dataset/exhibition2.sqlite", db_dir="crowd_dataset"):
        super(UAVEnvWrapper, self).__init__()
        
        # 从配置文件加载参数
        with open(config_path) as f:
            self.config = json.load(f)
        
        # 数据库目录和路径
        self.db_dir = db_dir
        self.db_path = db_path
        self.db_list = None
        self.current_db_index = 0
        self._update_db_list()
        
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
        
        # 数据库切换标志
        self.db_finished = False
        # 读取当前数据库的最大帧数
        self.max_frames = self._get_max_frame_from_db(self.db_path)

    def _get_max_frame_from_db(self, db_path):
        """从数据库中读取最大帧数"""
        try:
            # 连接数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 查询frame_data表中的最大frame值
            cursor.execute("SELECT MAX(frame) FROM frame_data")
            max_frame = cursor.fetchone()[0]
            
            # 关闭连接
            conn.close()
            
            # 如果找不到有效的最大帧，设置一个默认值
            if max_frame is None:
                print(f"警告: 在数据库 {db_path} 中未找到有效的帧数据，使用默认值3000")
                return 3000
                
            print(f"数据库 {db_path} 中的最大帧数: {max_frame}")
            return max_frame
            
        except Exception as e:
            print(f"从数据库读取最大帧数时出错: {e}，使用默认值3000")
            return 3000

    def _update_db_list(self):
        """更新数据库列表"""
        self.db_list = sorted(glob.glob(os.path.join(self.db_dir, "exhibition*.sqlite")))
        if not self.db_list:
            raise ValueError(f"在目录 {self.db_dir} 中未找到数据库文件")
        
        # 如果当前数据库不在列表中，将其添加到列表中
        if self.db_path not in self.db_list:
            self.db_list.append(self.db_path)
        
        # 设置当前数据库索引
        try:
            self.current_db_index = self.db_list.index(self.db_path)
        except ValueError:
            self.current_db_index = 0
            self.db_path = self.db_list[0]
        
        print(f"加载数据库列表: {self.db_list}")
        print(f"当前使用数据库: {self.db_path} (索引: {self.current_db_index})")

    def _switch_to_next_db(self):
        """切换到下一个数据库"""
        self.current_db_index = (self.current_db_index + 1) % len(self.db_list)
        self.db_path = self.db_list[self.current_db_index]
        self.db_finished = False
        # 更新最大帧数
        self.max_frames = self._get_max_frame_from_db(self.db_path)
        print(f"切换到新数据库: {self.db_path}，最大帧数: {self.max_frames}")
        return True

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
                - current_db: 当前使用的数据库
                - db_finished: 当前数据库是否已完成
                - max_frames: 当前数据库的最大帧数
        """
        # 将无人机位置展平为一维数组
        flat_uav_positions = self.uav_positions.flatten()
        
        # 检查full_density是否为None或为空
        if self.full_density is None:
            # 创建一个零矩阵作为替代
            full_density_with_channel = np.zeros((1, self.density_size, self.density_size), dtype=np.float32)
        else:
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
            "observed_ratio": np.float32(self.observed_ratio if self.observed_ratio is not None else 0.0),
            "observed_area_ratio": np.float32(self.observed_area_ratio if self.observed_area_ratio is not None else 0.0),
            "total_power": np.float32(self.total_power if self.total_power is not None else 0.0),
            "served_people": np.float32(self.served_people if self.served_people is not None else 0.0),
            "service_ratio": np.float32(self.service_ratio if self.service_ratio is not None else 0.0),
            "current_db": self.db_path,
            "db_finished": self.db_finished,
            "max_frames": self.max_frames
        }

    def _update_state(self):
        """更新环境状态和相关计算值"""
        try:
            # 计算帧ID
            self.frame_id = int(self.current_step * self.fps)
            
            # 检查是否超出当前数据库的最大帧数
            if self.frame_id >= self.max_frames:
                self.db_finished = True
                return
            
            # 获取完整的密度分布
            self.full_density = get_crowd_density(
                self.density_size,
                self.density_size,
                self.frame_id,
                db_path=self.db_path
            )
            
            # 检查从数据库获取的人群密度是否有效
            if self.full_density is None or np.sum(self.full_density) <= 0:
                print(f"警告: 在帧 {self.frame_id} 中未获取到有效的人群密度数据，可能已到达数据库末尾")
                self.db_finished = True
                return
            
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
        except Exception as e:
            print(f"更新环境状态时出错: {e}")
            # 标记数据库已完成
            self.db_finished = True
            
            # 设置合理的默认值，避免进一步的计算错误
            if not hasattr(self, 'frame_id') or self.frame_id is None:
                self.frame_id = int(self.current_step * self.fps)
                
            if not hasattr(self, 'full_density') or self.full_density is None:
                self.full_density = np.zeros((self.density_size, self.density_size))
                
            if not hasattr(self, 'observed_density') or self.observed_density is None:
                self.observed_density = np.zeros((self.density_size, self.density_size))
                
            if not hasattr(self, 'observation_mask') or self.observation_mask is None:
                self.observation_mask = np.zeros((self.density_size, self.density_size), dtype=bool)
                
            self.total_people = 0
            self.observed_people = 0
            self.observed_ratio = 0
            self.observed_area_ratio = 0

    def reset(self, seed=None, options=None):
        # 重置环境前检查数据库是否已完成
        if options is not None and options.get("switch_db", False):
            self._switch_to_next_db()
        elif self.db_finished:
            self._switch_to_next_db()
        
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
        
        # 重置数据库状态标志
        self.db_finished = False
        
        # 更新环境状态和缓存值
        self._update_state()
        
        # 如果更新后数据库标记为完成，则切换到下一个数据库并再次重置
        if self.db_finished:
            self._switch_to_next_db()
            return self.reset(seed=seed)
        
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
        
        # 检查数据库是否已完成
        if self.db_finished:
            # 设置终止标志
            terminated = True
            truncated = False
            
            # 设置最终奖励
            self.reward = 0.0
            
            # 返回结果，指示环境需要重置
            return self._get_obs(), self.reward, terminated, truncated, self._get_info()
        
        # 计算服务人数和能量消耗
        self.total_power, self.served_people = allocate_uav_service(
            self.frame_id, 
            self.uav_states,
            db_path=self.db_path
        )
        
        # 计算服务用户比例
        self.service_ratio = self.served_people / self.total_people if self.total_people > 0 else 0
        
        # 奖励计算
        decay_factor = np.exp(-0.001 * self.current_step)  # 指数衰减因子
        epsilon = 1e-6  # 防止除零
        
        # 特殊情况：如果没有人群，给予中性奖励
        if self.total_people <= 0:
            self.reward = 0.0
            # 检查终止条件
            terminated = self.current_step >= self.max_steps
            truncated = False  # 可根据需要添加其他终止条件
            return self._get_obs(), self.reward, terminated, truncated, self._get_info()
        
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
        # 增加额外检查，避免除数过小
        denominator = (self.service_ratio**2) * self.total_people + epsilon
        if denominator > epsilon:  # 确保分母有意义
            power_penalty = power_penalty_weight * (self.total_power / denominator)
        else:
            # 如果分母太小，使用一个备选公式
            power_penalty = power_penalty_weight * self.total_power
        
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
