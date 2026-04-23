"""
vibsignal.utils — 工具函数模块

提供以下功能：
    generate_bearing_signal   生成含轴承故障的仿真信号
    generate_gear_signal      生成含齿轮故障的仿真信号
    generate_shaft_signal     生成转轴振动仿真信号（不平衡、不对中等）
    next_power_of_2           最近的 2 的幂次
    db                        线性幅值转 dB
    rms_to_peak               RMS → 峰值（正弦信号：× √2）
    peak_to_rms               峰值 → RMS
    resample_signal           重采样（整数倍降采样/升采样）
    unit_convert              传感器信号单位换算（加速度↔速度↔位移）
    segment_signal            按帧分割信号
"""

import numpy as np
from typing import Optional, Tuple


# ── 信号生成 ──────────────────────────────────────────────────────────────────

def generate_bearing_signal(fs: float = 20000,
                             duration: float = 1.0,
                             rpm: float = 1500,
                             fault_type: str = 'BPFO',
                             n_balls: int = 9,
                             d_ball: float = 7.0,
                             D_pitch: float = 38.0,
                             snr_db: float = 10.0,
                             rng_seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    生成含轴承故障的仿真加速度信号。

    模型：低频旋转 + 共振频带调制 + 故障冲击脉冲 + 高斯白噪声。

    Parameters
    ----------
    fs         : 采样频率 (Hz)，默认 20000
    duration   : 信号时长 (s)，默认 1.0
    rpm        : 转速 (r/min)，默认 1500
    fault_type : 故障类型 'BPFO'|'BPFI'|'BSF'|'FTF'，默认 'BPFO'
    n_balls    : 滚动体数，默认 9
    d_ball     : 滚动体直径 (mm)，默认 7.0
    D_pitch    : 节径 (mm)，默认 38.0
    snr_db     : 信噪比 (dB)，默认 10
    rng_seed   : 随机种子，默认 42

    Returns
    -------
    signal : np.ndarray，仿真信号
    fs     : float，采样频率
    """
    from . import bearing as brg
    rng = np.random.default_rng(rng_seed)
    t = np.arange(int(fs * duration)) / fs
    N = len(t)
    fr = rpm / 60.0

    ffreqs = brg.fault_frequencies(rpm, n_balls, d_ball, D_pitch)
    fault_freq = ffreqs[fault_type]

    # 旋转基频成分
    sig = 0.5 * np.sin(2 * np.pi * fr * t) + \
          0.2 * np.sin(2 * np.pi * 2 * fr * t + 0.5)

    # 共振频率（模拟轴承结构共振）
    f_res = 3500.0
    decay = 200.0     # 衰减系数

    # 故障冲击脉冲（每隔 1/fault_freq 产生一次衰减振荡）
    period_samples = int(round(fs / fault_freq))
    impulse_train = np.zeros(N)
    for k in range(0, N, period_samples):
        width = min(int(fs / decay * 5), N - k)
        idx = np.arange(width)
        impulse = np.exp(-decay / fs * idx) * np.sin(2 * np.pi * f_res / fs * idx)
        impulse_train[k:k + width] += impulse

    sig += impulse_train

    # 加噪
    signal_power = np.mean(sig ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    sig += rng.normal(0, np.sqrt(noise_power), N)

    return sig.astype(np.float32), fs


def generate_gear_signal(fs: float = 10000,
                          duration: float = 1.0,
                          rpm: float = 1500,
                          n_teeth: int = 20,
                          fault_type: str = 'broken_tooth',
                          snr_db: float = 15.0,
                          rng_seed: int = 0) -> Tuple[np.ndarray, float]:
    """
    生成含齿轮故障的仿真加速度信号。

    模型：啮合谐波 + 幅值/频率调制（故障边带）+ 白噪声。

    Parameters
    ----------
    fs         : 采样频率 (Hz)
    duration   : 信号时长 (s)
    rpm        : 转速 (r/min)
    n_teeth    : 齿轮齿数
    fault_type : 'normal'|'broken_tooth'|'wear'，默认 'broken_tooth'
    snr_db     : 信噪比 (dB)
    rng_seed   : 随机种子

    Returns
    -------
    signal : np.ndarray，仿真信号
    fs     : float
    """
    rng = np.random.default_rng(rng_seed)
    t = np.arange(int(fs * duration)) / fs
    N = len(t)
    fr = rpm / 60.0
    fm = fr * n_teeth        # 啮合频率

    # 正常啮合谐波
    sig = np.zeros(N)
    for k in range(1, 5):
        sig += (1.0 / k) * np.sin(2 * np.pi * fm * k * t + rng.uniform(0, np.pi))

    # 旋转基频
    sig += 0.3 * np.sin(2 * np.pi * fr * t)

    # 故障调制
    if fault_type == 'broken_tooth':
        # 断齿：每转一圈产生一次脉冲式幅值调制
        modulation = 1.0 + 1.5 * np.abs(np.sin(np.pi * fr * t)) ** 10
        sig *= modulation
    elif fault_type == 'wear':
        # 均匀磨损：整体幅值调制（边带均匀分布）
        modulation = 1.0 + 0.3 * np.cos(2 * np.pi * fr * t)
        sig *= modulation

    # 加噪
    signal_power = np.mean(sig ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    sig += rng.normal(0, np.sqrt(noise_power), N)

    return sig.astype(np.float32), fs


def generate_shaft_signal(fs: float = 5000,
                           duration: float = 1.0,
                           rpm: float = 3000,
                           fault_type: str = 'unbalance',
                           snr_db: float = 20.0,
                           rng_seed: int = 7) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    生成转轴双通道位移仿真信号（用于轴心轨迹分析）。

    Parameters
    ----------
    fs         : 采样频率 (Hz)
    duration   : 信号时长 (s)
    rpm        : 转速 (r/min)
    fault_type : 'normal'|'unbalance'|'misalignment'|'rub'，默认 'unbalance'
    snr_db     : 信噪比
    rng_seed   : 随机种子

    Returns
    -------
    x, y : 两路正交位移信号 (µm)
    fs   : 采样频率
    """
    rng = np.random.default_rng(rng_seed)
    t = np.arange(int(fs * duration)) / fs
    N = len(t)
    fr = rpm / 60.0
    omega = 2 * np.pi * fr

    if fault_type == 'normal':
        x = 30 * np.cos(omega * t)
        y = 30 * np.sin(omega * t)

    elif fault_type == 'unbalance':
        # 不平衡：1X 主导，轨迹近似圆
        x = 60 * np.cos(omega * t) + 5 * np.cos(2 * omega * t)
        y = 60 * np.sin(omega * t) + 5 * np.sin(2 * omega * t)

    elif fault_type == 'misalignment':
        # 不对中：2X 分量显著，轨迹呈"8"字形
        x = 30 * np.cos(omega * t) + 25 * np.cos(2 * omega * t)
        y = 30 * np.sin(omega * t) - 20 * np.sin(2 * omega * t)

    elif fault_type == 'rub':
        # 碰摩：1X 主导 + 高次谐波 + 直流偏置
        x = 25 * np.cos(omega * t) + \
            8 * np.cos(2 * omega * t) + \
            4 * np.cos(3 * omega * t) + 10
        y = 25 * np.sin(omega * t) + \
            6 * np.sin(2 * omega * t) + \
            3 * np.sin(3 * omega * t) + 8
    else:
        raise ValueError(f"未知故障类型: {fault_type}")

    # 加噪
    sp = np.mean(x ** 2)
    np_var = sp / (10 ** (snr_db / 10))
    x += rng.normal(0, np.sqrt(np_var), N)
    y += rng.normal(0, np.sqrt(np_var), N)

    return x.astype(np.float32), y.astype(np.float32), fs


# ── 数学工具 ──────────────────────────────────────────────────────────────────

def next_power_of_2(n: int) -> int:
    """返回 ≥ n 的最小 2 的幂次。"""
    return int(2 ** np.ceil(np.log2(n)))


def db(amplitude: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """幅值转 dB：20 * log10(amplitude / ref)。"""
    return 20 * np.log10(np.asarray(amplitude, float) / ref + 1e-12)


def rms_to_peak(rms_val: float) -> float:
    """RMS → 峰值（纯正弦：× √2）。"""
    return rms_val * np.sqrt(2)


def peak_to_rms(peak_val: float) -> float:
    """峰值 → RMS（纯正弦：÷ √2）。"""
    return peak_val / np.sqrt(2)


def resample_signal(data: np.ndarray, fs: float,
                    target_fs: float) -> Tuple[np.ndarray, float]:
    """
    整数比重采样（用 scipy.signal.resample_poly）。

    Parameters
    ----------
    data      : 输入信号
    fs        : 原始采样频率 (Hz)
    target_fs : 目标采样频率 (Hz)

    Returns
    -------
    resampled : np.ndarray
    target_fs : float
    """
    from scipy import signal as sp_signal
    from math import gcd
    fs_i = int(round(fs))
    fs_t = int(round(target_fs))
    g = gcd(fs_i, fs_t)
    up, down = fs_t // g, fs_i // g
    resampled = sp_signal.resample_poly(data, up, down)
    return resampled, target_fs


def unit_convert(data: np.ndarray, fs: float,
                 from_unit: str, to_unit: str) -> np.ndarray:
    """
    传感器信号单位换算（数值积分 / 微分）。

    支持转换方向：
        'acceleration' → 'velocity'   : 积分（时域累加 × dt）
        'velocity'     → 'displacement': 积分
        'acceleration' → 'displacement': 积分两次
        'displacement' → 'velocity'   : 微分
        'velocity'     → 'acceleration': 微分

    Parameters
    ----------
    data      : 输入信号
    fs        : 采样频率
    from_unit : 源单位 'acceleration'|'velocity'|'displacement'
    to_unit   : 目标单位

    Returns
    -------
    np.ndarray，换算后信号
    """
    dt = 1.0 / fs
    order_map = {'displacement': 0, 'velocity': 1, 'acceleration': 2}
    if from_unit not in order_map or to_unit not in order_map:
        raise ValueError(f"未知单位: {from_unit} 或 {to_unit}")
    diff = order_map[from_unit] - order_map[to_unit]
    result = data.copy().astype(float)
    if diff < 0:
        # 需要微分
        for _ in range(-diff):
            result = np.gradient(result, dt)
    elif diff > 0:
        # 需要积分（梯形法，去直流）
        from scipy.integrate import cumulative_trapezoid
        for _ in range(diff):
            result = cumulative_trapezoid(result, dx=dt, initial=0)
            result -= np.mean(result)
    return result


def segment_signal(data: np.ndarray, frame_size: int,
                   hop_size: int = None,
                   window: bool = False) -> np.ndarray:
    """
    将信号分割为等长帧（用于批量特征提取）。

    Parameters
    ----------
    data       : 输入信号
    frame_size : 每帧长度（样本数）
    hop_size   : 帧移长度；None 时等于 frame_size（不重叠）
    window     : 是否对每帧加 Hann 窗，默认 False

    Returns
    -------
    frames : np.ndarray，shape = (n_frames, frame_size)
    """
    if hop_size is None:
        hop_size = frame_size
    N = len(data)
    starts = range(0, N - frame_size + 1, hop_size)
    frames = np.array([data[s:s + frame_size] for s in starts])
    if window:
        win = np.hanning(frame_size)
        frames = frames * win[np.newaxis, :]
    return frames
