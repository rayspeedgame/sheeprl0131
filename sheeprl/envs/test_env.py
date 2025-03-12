import numpy as np
import matplotlib.pyplot as plt
from sheeprl.envs.UAVenv import UAVEnvWrapper

def test_env():
    # 创建环境
    env = UAVEnvWrapper()
    print("环境创建成功")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 重置环境
    obs, _ = env.reset()
    print("\n初始观测:")
    print(f"无人机位置: {obs['uav_positions']}")
    print(f"密度矩阵形状: {obs['density_matrix'].shape}")
    print(f"密度矩阵非零元素数量: {np.count_nonzero(obs['density_matrix'])}")
    
    # 执行随机动作序列
    print("\n执行随机动作测试:")
    total_reward = 0
    n_steps = 10
    
    for step in range(n_steps):
        # 生成随机动作
        action = env.action_space.sample()
        
        # 执行步骤
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\n步骤 {step+1}:")
        print(f"动作: 平均移动量 {action.mean(axis=1)}")
        print(f"无人机位置: {obs['uav_positions']}")
        print(f"密度矩阵形状: {obs['density_matrix'].shape}")
        print(f"密度矩阵非零元素数量: {np.count_nonzero(obs['density_matrix'])}")
        print(f"奖励: {reward}")
        
        if terminated or truncated:
            print("环境终止")
            break
    
    print(f"\n测试完成. 总奖励: {total_reward}")
    
    # 可视化最后一帧的密度矩阵
    plt.figure(figsize=(10, 6))
    plt.imshow(obs['density_matrix'], cmap='viridis')
    plt.colorbar(label='人群密度')
    plt.title('观测到的人群密度矩阵')
    
    # 在密度矩阵上标记无人机位置
    for i, pos in enumerate(obs['uav_positions']):
        # 转换无人机位置到网格坐标
        x_max, y_max = env.boundary[1][0], env.boundary[1][1]
        grid_x = int(pos[0] / x_max * (env.density_size - 1))
        grid_y = int((y_max - pos[1]) / y_max * (env.density_size - 1))
        
        # 确保坐标在有效范围内
        grid_x = max(0, min(grid_x, env.density_size - 1))
        grid_y = max(0, min(grid_y, env.density_size - 1))
        
        plt.plot(grid_x, grid_y, 'r*', markersize=10, label=f'UAV {i+1}' if i==0 else "")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('density_visualization.png')
    plt.show()
    
    return env, obs

if __name__ == "__main__":
    env, last_obs = test_env() 