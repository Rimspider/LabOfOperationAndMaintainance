import os
import numpy as np
from scipy import stats, signal
from scipy.signal import hilbert

data_slice=slice(15, 15000)
N=14985
T=1.0/2000.0

def get_time_domain_features(x):
    x = np.asarray(x, dtype=float)  # Ensure x is a numeric array
    features = {
        '均值': np.mean(x),
        '标准差': np.std(x),
        '方差': np.var(x),
        '均方根(RMS)': np.sqrt(np.mean(x**2)),
        '峰值': np.max(np.abs(x)),
        '峰峰值': np.ptp(x),
        '偏度': stats.skew(x),      # 不对称性
        '峭度': stats.kurtosis(x),   # 冲击性
        '峰值因子': np.max(np.abs(x)) / np.sqrt(np.mean(x**2)),
        '波形因子': np.sqrt(np.mean(x**2)) / np.mean(np.abs(x)),
        '脉冲因子': np.max(np.abs(x)) / np.mean(np.abs(x)),
        '裕度因子': np.max(np.abs(x)) / np.mean(np.sqrt(np.abs(x)))**2
    }
    for key, value in features.items():
        print(f"{key}: {value:.4f}")
    return features

def hilbert_transform(x):
    """
    计算信号的希尔伯特变换
    
    参数:
    x: 输入信号 (一维数组)
    
    返回:
    analytic_signal: 解析信号 (实部为原始信号，虚部为希尔伯特变换)
    amplitude: 瞬时幅值
    phase: 瞬时相位
    frequency: 瞬时频率
    """
    analytic_signal = hilbert(x)
    amplitude = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    
    # 计算瞬时频率（去除第一个和最后一个点）
    frequency = np.diff(phase) / (2.0 * np.pi)
    frequency = np.append(frequency[0], frequency)
    
    return analytic_signal, amplitude, phase, frequency