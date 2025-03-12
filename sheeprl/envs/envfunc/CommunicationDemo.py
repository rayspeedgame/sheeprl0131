import sheeprl.envs.envfunc.Communication as comm
import numpy as np

# 计算在以下条件下所需的最小发射功率：
distance = 10  # 100米
frequency = 2.4  # 2.4GHz
bandwidth = 20  # 20MHz
data_rate = 100000  # 1000kbps
noise_power = -50  # -90dBm
tx_gain = 2  # 2dB
rx_gain = 2  # 2dB

power = comm.calculate_required_power(distance, frequency, bandwidth, data_rate, noise_power, tx_gain, rx_gain)
print(f"Required transmission power: {power*1000:.2f} mW")