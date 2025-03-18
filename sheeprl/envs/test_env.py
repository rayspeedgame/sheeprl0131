import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sheeprl.envs.UAVenv import UAVEnvWrapper

def test_env():
    # 创建环境
    env = UAVEnvWrapper()
    print("环境创建成功")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 重置环境
    obs, info = env.reset()
    print("\n初始观测:")
    print(f"观测密度矩阵形状: {obs['density_matrix'].shape}")
    print(f"密度矩阵非零元素数量: {np.count_nonzero(obs['density_matrix'])}")
    print(f"无人机位置: {info['uav_positions']}")
    print(f"观测到的人群比例: {info['observed_ratio']:.4f}")
    print(f"观测范围比例: {info['observed_area_ratio']:.4f}")
    
    # 执行随机动作序列
    print("\n执行随机动作测试:")
    total_reward = 0
    n_steps = 10
    
    # 记录每一步的信息用于分析
    history = {
        'rewards': [],
        'observed_ratios': [],
        'observed_area_ratios': [],
        'uav_positions': []
    }
    
    for step in range(n_steps):
        # 生成随机动作
        action = env.action_space.sample()
        
        # 执行步骤
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 记录历史数据
        history['rewards'].append(reward)
        history['observed_ratios'].append(info['observed_ratio'])
        history['observed_area_ratios'].append(info['observed_area_ratio'])
        history['uav_positions'].append(info['uav_positions'].copy())
        
        print(f"\n步骤 {step+1}:")
        print(f"动作: 平均移动量 {np.mean(action, axis=0)}")
        print(f"无人机位置: {info['uav_positions']}")
        print(f"密度矩阵形状: {obs['density_matrix'].shape}")
        print(f"密度矩阵非零元素数量: {np.count_nonzero(obs['density_matrix'])}")
        print(f"观测到的人群比例: {info['observed_ratio']:.4f}")
        print(f"观测范围比例: {info['observed_area_ratio']:.4f}")
        print(f"奖励: {reward:.4f}")
        
        if terminated or truncated:
            print("环境终止")
            break
    
    print(f"\n测试完成. 总奖励: {total_reward:.4f}")
    
    # 创建可视化图表
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 可视化最后一帧的密度矩阵
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(obs['density_matrix'], cmap='viridis')
    plt.colorbar(im, ax=ax1, label='人群密度')
    ax1.set_title('观测到的人群密度矩阵')
    
    # 在密度矩阵上标记无人机位置
    uav_positions = info['uav_positions']
    x_min, y_min = env.boundary[0][0], env.boundary[0][1]
    x_max, y_max = env.boundary[1][0], env.boundary[1][1]
    
    for i, pos in enumerate(uav_positions):
        # 转换无人机位置到网格坐标
        grid_x = int((pos[0] - x_min) / (x_max - x_min) * (env.density_size - 1))
        grid_y = int((y_max - pos[1]) / (y_max - y_min) * (env.density_size - 1))
        
        # 确保坐标在有效范围内
        grid_x = max(0, min(grid_x, env.density_size - 1))
        grid_y = max(0, min(grid_y, env.density_size - 1))
        
        ax1.plot(grid_x, grid_y, 'r*', markersize=10, label=f'UAV {i+1}' if i==0 else "")
    
    # 2. 可视化完整的人群密度矩阵（从info中获取）
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(info['full_density'], cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='人群密度')
    ax2.set_title('完整的人群密度矩阵')
    
    # 同样标记无人机位置和观测范围
    for i, pos in enumerate(uav_positions):
        # 转换无人机位置到网格坐标
        grid_x = int((pos[0] - x_min) / (x_max - x_min) * (env.density_size - 1))
        grid_y = int((y_max - pos[1]) / (y_max - y_min) * (env.density_size - 1))
        
        # 确保坐标在有效范围内
        grid_x = max(0, min(grid_x, env.density_size - 1))
        grid_y = max(0, min(grid_y, env.density_size - 1))
        
        ax2.plot(grid_x, grid_y, 'r*', markersize=10)
        
        # 估计观测半径
        z = pos[2]  # 高度
        observation_radius = z * np.tan(np.radians(60))  # 假设flare_angle=60度
        radius_in_grid = int(observation_radius / (x_max - x_min) * env.density_size)
        
        # 绘制观测圆
        observation_circle = Circle((grid_x, grid_y), radius_in_grid, 
                                    fill=False, edgecolor='r', linestyle='--')
        ax2.add_patch(observation_circle)
    
    # 3. 绘制观测比例变化曲线
    ax3 = fig.add_subplot(2, 2, 3)
    steps = np.arange(1, len(history['observed_ratios'])+1)
    ax3.plot(steps, history['observed_ratios'], 'b-', label='人群观测比例')
    ax3.plot(steps, history['observed_area_ratios'], 'g--', label='区域观测比例')
    ax3.set_xlabel('步骤')
    ax3.set_ylabel('比例')
    ax3.set_title('观测比例变化')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 绘制奖励变化曲线
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(steps, history['rewards'], 'r-')
    ax4.set_xlabel('步骤')
    ax4.set_ylabel('奖励')
    ax4.set_title('奖励变化')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('uav_env_test_visualization.png')
    plt.show()
    
    return env, obs, info, history

if __name__ == "__main__":
    env, last_obs, last_info, history = test_env() 