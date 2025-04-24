import numpy as np
from sheeprl.envs.envfunc.Crowd import get_positions, get_crowd_density
from sheeprl.envs.envfunc.Communication import calculate_required_power
import sqlite3

def allocate_uav_service(frame_id, uav_states, max_power=1.0, db_path='exhibition.sqlite'):
    """
    分配无人机服务并计算总功率消耗
    
    参数:
        frame_id (int): 帧编号
        uav_states (numpy.ndarray): 无人机状态矩阵 [uav_id, x, y, z]
        max_power (float): 无人机最大发射功率，单位：瓦特
        db_path (str): 数据库路径
        
    返回:
        tuple: (总功率消耗, 服务成功的用户数)
    """
    # 通信参数设置
    FREQUENCY = 3.5  # GHz
    BANDWIDTH = 3.6   # MHz
    DATA_RATE = 10000 # kbps
    NOISE_POWER = -50 # dBm
    TX_GAIN = 21      # dB
    RX_GAIN = 2      # dB
    
    # 1. 获取游客位置信息
    visitor_positions = get_positions(frame_id, db_path=db_path)
    
    # 检查visitor_positions是否为空或一维数组
    if len(visitor_positions) == 0 or visitor_positions.ndim == 1:
        # 返回零功率和零服务人数
        return 0.0, 0
    
    visitor_status = np.zeros((len(visitor_positions), 4))  # [visitor_id, x, y, served_flag]
    visitor_status[:, 0] = visitor_positions[:, 0]  # ID
    visitor_status[:, 1:3] = visitor_positions[:, 1:3]  # x, y coordinates
    visitor_status[:, 3] = 0  # 初始化服务状态为0（未服务）
    
    # 2. 使用无人机当前状态（不需要更新位置）
    current_uav_states = uav_states.astype(np.float64)
    
    # 3. 预先计算所有无人机到所有游客的距离
    n_uavs = len(current_uav_states)
    n_visitors = len(visitor_positions)
    distance_matrix = np.zeros((n_uavs, n_visitors))
    
    for i in range(n_uavs):
        uav_pos = current_uav_states[i, 1:4]
        visitor_pos = np.column_stack((visitor_status[:, 1:3], np.zeros(n_visitors)))
        distance_matrix[i] = np.linalg.norm(uav_pos - visitor_pos, axis=1)
    
    # 为每个无人机创建按距离排序的游客索引
    sorted_visitor_indices = np.argsort(distance_matrix, axis=1)
    
    # 创建无人机服务状态矩阵 [uav_id, x, y, z, current_power, is_full]
    uav_service_status = np.zeros((n_uavs, 6))
    uav_service_status[:, :4] = current_uav_states  # 复制位置信息
    uav_service_status[:, 4] = 0  # 初始化当前功率为0
    uav_service_status[:, 5] = 0  # 初始化未满员状态
    
    total_power = 0
    served_users = 0
    
    while not np.all(uav_service_status[:, 5] == 1):
        # 检查是否所有用户都已被服务
        if served_users == n_visitors:
            break
            
        # 找到未满员且功率最低的无人机
        available_mask = uav_service_status[:, 5] == 0
        if not np.any(available_mask):
            break
            
        available_powers = np.where(available_mask, uav_service_status[:, 4], np.inf)
        current_uav_idx = np.argmin(available_powers)
        
        # 遍历当前无人机的排序游客列表，找到未服务的最近游客
        for visitor_idx in sorted_visitor_indices[current_uav_idx]:
            if visitor_status[visitor_idx, 3] == 0:  # 未服务
                distance = distance_matrix[current_uav_idx, visitor_idx]
                required_power = calculate_required_power(
                    distance, FREQUENCY, BANDWIDTH, DATA_RATE,
                    NOISE_POWER, TX_GAIN, RX_GAIN
                )
                
                if uav_service_status[current_uav_idx, 4] + required_power <= max_power:
                    # 更新无人机当前功率
                    uav_service_status[current_uav_idx, 4] += required_power
                    total_power += required_power
                    
                    # 更新用户服务状态
                    visitor_status[visitor_idx, 3] = 1
                    served_users += 1
                    break
                else:
                    # 如果最近可服务的用户功率超过最大功率，标记无人机为满员
                    uav_service_status[current_uav_idx, 5] = 1
                    break
            
    return total_power, served_users

def get_observed_density(frame_id, uav_positions, grid_rows=10, grid_cols=10, flare_angle=60, db_path='exhibition.sqlite'):
    """
    计算无人机观测到的人群密度信息
    
    参数:
        frame_id (int): 帧编号
        uav_positions (numpy.ndarray): 无人机位置矩阵 [uav_id, x, y, z]
        grid_rows (int): 网格行数
        grid_cols (int): 网格列数
        flare_angle (float): 无人机观测角度（度），默认60度
        db_path (str): 数据库路径
        
    返回:
        tuple: (observed_density, observation_mask)
            - observed_density: 观测到的人群密度矩阵 (grid_rows x grid_cols)
            - observation_mask: 观测掩码，指示哪些区域被观测到 (grid_rows x grid_cols)
    """
    # 获取人群密度信息
    density_matrix = get_crowd_density(grid_cols, grid_rows, frame_id, db_path=db_path)
    
    # 创建观测掩码矩阵（标记哪些网格可以被观测到）
    observation_mask = np.zeros((grid_rows, grid_cols), dtype=bool)
    
    # 检查无人机位置是否有效
    if uav_positions is None or len(uav_positions) == 0:
        # 如果无人机位置数据无效，返回全零密度和掩码
        return np.zeros((grid_rows, grid_cols)), observation_mask
    
    # 确保uav_positions至少有4列（id, x, y, z）
    if uav_positions.shape[1] < 4:
        # 如果列数不足，返回全零密度和掩码
        return np.zeros((grid_rows, grid_cols)), observation_mask
    
    # 获取场地范围（从metadata表）
    try:
        conn = sqlite3.connect(db_path)
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
    except Exception as e:
        # 如果数据库查询出错，返回全零密度和掩码
        print(f"查询元数据出错: {e}")
        return np.zeros((grid_rows, grid_cols)), observation_mask
    
    # 计算每个网格的大小
    grid_width = (max_x - min_x) / grid_cols
    grid_height = (max_y - min_y) / grid_rows
    
    # 计算每个网格的中心点坐标（注意y轴方向是从下到上）
    x_centers = np.linspace(min_x + grid_width/2, max_x - grid_width/2, grid_cols)
    y_centers = np.linspace(min_y + grid_height/2, max_y - grid_height/2, grid_rows)
    
    # 将角度转换为弧度
    flare_angle_rad = np.radians(flare_angle)
    
    # 对每个无人机
    for uav in uav_positions:
        try:
            uav_x, uav_y, uav_z = uav[1:4]
            # 确保高度是正值
            if uav_z <= 0:
                continue
                
            # 计算观测半径（平面距离）
            observation_radius = uav_z * np.tan(flare_angle_rad)
            
            # 检查每个网格中心点是否在观测范围内
            for i, y in enumerate(y_centers):
                for j, x in enumerate(x_centers):
                    # 计算平面距离
                    plane_distance = np.sqrt((x - uav_x)**2 + (y - uav_y)**2)
                    if plane_distance <= observation_radius:
                        # 注意：i是从下到上的行索引
                        observation_mask[grid_rows-1-i, j] = True
        except (IndexError, ValueError) as e:
            print(f"处理无人机数据时出错: {e}")
            continue
    
    # 将未观测到的区域密度设为0
    observed_density = np.where(observation_mask, density_matrix, 0)
    
    return observed_density, observation_mask
