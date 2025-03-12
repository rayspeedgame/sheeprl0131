import numpy as np
from sheeprl.envs.envfunc.UAV import allocate_uav_service, get_observed_density
import matplotlib.pyplot as plt
import sqlite3
from sheeprl.envs.envfunc.Crowd import get_crowd_density


# 创建测试数据，使用float类型
frame_id = 1000
uav_movement = np.array([
    [1, 0.1, 0.2, 0.0],  # UAV 1的移动量
    [2, -0.1, 0.1, 0.0]  # UAV 2的移动量
], dtype=np.float64)

uav_states = np.array([
    [1, 0.0, 0.0, 0.0],  # UAV 1的当前状态
    [2, 0.0, 0.0, 0.0]   # UAV 2的当前状态
], dtype=np.float64)

uav_init_positions = np.array([
    [1, 10, 10, 30],  # UAV 1的初始位置
    [2, 20, 20, 30]   # UAV 2的初始位置
])

# 运行分配
total_power, served_count = allocate_uav_service(
    frame_id, uav_movement, uav_states
)
print(f"总功率消耗: {total_power:.2f}W")
print(f"服务用户数: {served_count}")

# 无人机位置矩阵 [uav_id, x, y, z]
uav_positions = np.array([
    [1, 10.0, 10.0, 30.0],  # UAV 1
    [2, 100.0, 20.0, 30.0]   # UAV 2
])

# 获取场地范围（从metadata表）
conn = sqlite3.connect('exhibition.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT value FROM metadata WHERE key = 'xmin'")
min_x = float(cursor.fetchone()[0])
cursor.execute("SELECT value FROM metadata WHERE key = 'xmax'")
max_x = float(cursor.fetchone()[0])
cursor.execute("SELECT value FROM metadata WHERE key = 'ymin'")
min_y = float(cursor.fetchone()[0])
cursor.execute("SELECT value FROM metadata WHERE key = 'ymax'")
max_y = float(cursor.fetchone()[0])
conn.close()

# 获取原始密度矩阵和观测密度矩阵
original_density = get_crowd_density(grid_cols=10, grid_rows=10, frame_id=1000)
observed_density = get_observed_density(
    frame_id=1000,
    uav_positions=uav_positions,
    flare_angle=20
)

# 创建并排显示的图表
plt.figure(figsize=(20, 8))

# 显示原始密度图
plt.subplot(1, 2, 1)
plt.imshow(original_density, cmap='YlOrRd', extent=[min_x, max_x, min_y, max_y])
plt.colorbar(label='人群密度')
plt.title('原始人群密度分布')
plt.xlabel('X坐标 (m)')
plt.ylabel('Y坐标 (m)')
plt.grid(True)

# 显示观测密度图
plt.subplot(1, 2, 2)
plt.imshow(observed_density, cmap='YlOrRd', extent=[min_x, max_x, min_y, max_y])
plt.colorbar(label='人群密度')

# 标记无人机位置
for uav in uav_positions:
    plt.plot(uav[1], uav[2], 'b^', markersize=10, label=f'UAV {int(uav[0])}')

plt.title('无人机观测到的人群密度分布')
plt.xlabel('X坐标 (m)')
plt.ylabel('Y坐标 (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()  # 自动调整子图布局
plt.show()

# 打印统计信息
print("\n密度统计信息:")
print(f"原始最大密度: {np.max(original_density):.2f}")
print(f"原始平均密度: {np.mean(original_density):.2f}")
print(f"观测最大密度: {np.max(observed_density):.2f}")
print(f"观测平均密度: {np.mean(observed_density):.2f}")
print(f"观测到的网格数: {np.sum(observed_density > 0)}")
print(f"总网格数: {observed_density.size}")