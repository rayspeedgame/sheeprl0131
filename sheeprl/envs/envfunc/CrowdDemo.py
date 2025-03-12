import sqlite3
import numpy as np
import sheeprl.envs.envfunc.Crowd as Crowd


density = Crowd.get_crowd_density(10, 10, 3000)
print(density)

# 获取第2000帧的所有游客位置
positions = Crowd.get_positions(2000)
# 访问第一个游客的信息
visitor_id = positions[0][0]
x = positions[0][1]
y = positions[0][2]
print(visitor_id, x, y)



