import numpy as np


def calculate_required_power(distance, frequency, bandwidth, data_rate, noise_power, tx_gain, rx_gain):
    """
    计算满足传输速率要求的最小发射功率
    
    参数:
        distance (float): 设备间距离，单位：米
        frequency (float): 工作频率，单位：GHz
        bandwidth (float): 信道带宽，单位：MHz
        data_rate (float): 最小需求传输速率，单位：kbps
        noise_power (float): 噪声功率，单位：dBm
        tx_gain (float): 发送天线增益，单位：dB
        rx_gain (float): 接收天线增益，单位：dB
    
    返回:
        float: 所需最小发射功率，单位：瓦特
    """
    # 常量定义
    C = 3e8  # 光速，单位：m/s
    
    # 单位转换
    frequency = frequency * 1e9  # GHz to Hz
    bandwidth = bandwidth * 1e6  # MHz to Hz
    data_rate = data_rate * 1e3  # kbps to bps
    noise_power = 10 ** ((noise_power - 30) / 10)  # dBm to W
    
    # 计算自由空间路径损耗 (FSPL)
    fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
    
    # 计算总路径损耗（考虑天线增益）
    path_loss_db = fspl_db - tx_gain - rx_gain
    path_loss = 10 ** (path_loss_db / 10)
    
    # 根据香农公式计算所需的接收信噪比
    # C = B * log2(1 + SNR)
    required_snr = 2 ** (data_rate / bandwidth) - 1
    
    # 计算所需的发射功率
    required_power = required_snr * noise_power * path_loss
    
    return required_power

