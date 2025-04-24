import sqlite3
import numpy as np

def get_crowd_density(grid_cols, grid_rows, frame_id, db_path='exhibition.sqlite'):
    """
    计算指定帧下各网格的人群密度
    
    参数:
        grid_cols (int): 网格列数
        grid_rows (int): 网格行数
        frame_id (int): 帧编号
        db_path (str): 数据库路径
    
    返回:
        numpy.ndarray: 人群密度矩阵 (grid_rows x grid_cols)，原点在左上角，y轴向上
    """
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 从metadata表获取坐标范围，使用key-value格式查询
    cursor.execute("SELECT value FROM metadata WHERE key = 'xmin'")
    min_x = float(cursor.fetchone()[0])
    cursor.execute("SELECT value FROM metadata WHERE key = 'xmax'")
    max_x = float(cursor.fetchone()[0])
    cursor.execute("SELECT value FROM metadata WHERE key = 'ymin'")
    min_y = float(cursor.fetchone()[0])
    cursor.execute("SELECT value FROM metadata WHERE key = 'ymax'")
    max_y = float(cursor.fetchone()[0])
    
    # 计算每个网格的大小
    grid_width = (max_x - min_x) / grid_cols
    grid_height = (max_y - min_y) / grid_rows
    
    # 初始化密度矩阵
    density_matrix = np.zeros((grid_rows, grid_cols))
    
    # 获取指定帧的所有人员位置
    cursor.execute("""
        SELECT pos_x, pos_y 
        FROM trajectory_data 
        WHERE frame = ?
    """, (frame_id,))

    positions = cursor.fetchall()
    
    # 统计每个网格中的人数
    for x, y in positions:
        # 计算位置所在的网格索引
        col_idx = int((x - min_x) // grid_width)
        row_idx = int((max_y - y) // grid_height)
        
        # 确保索引在有效范围内
        if 0 <= col_idx < grid_cols and 0 <= row_idx < grid_rows:
            density_matrix[row_idx][col_idx] += 1
    
    # 关闭数据库连接
    conn.close()
    
    return density_matrix

def get_positions(frame_id, db_path='exhibition.sqlite'):
    """
    获取指定帧中所有游客的精确位置信息
    
    参数:
        frame_id (int): 帧编号
        db_path (str): 数据库路径
    
    返回:
        numpy.ndarray: 位置信息矩阵，每行包含 [visitor_id, x, y]
    """
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取指定帧的所有游客位置
    cursor.execute("""
        SELECT id, pos_x, pos_y 
        FROM trajectory_data 
        WHERE frame = ?
        ORDER BY id
    """, (frame_id,))
    
    # 获取查询结果
    results = cursor.fetchall()
    
    # 关闭数据库连接
    conn.close()
    
    # 如果结果为空，返回一个空的二维数组，形状为(0,3)
    if not results:
        return np.zeros((0, 3))
    
    # 将结果转换为numpy数组
    positions = np.array(results)
    
    return positions


