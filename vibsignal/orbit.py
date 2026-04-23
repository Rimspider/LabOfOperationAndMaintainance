"""
vibsignal.orbit — 轴心轨迹分析模块（用于转轴振动监测）

提供以下功能：
    orbit_plot_data      计算轴心轨迹（x-y 平面位移）
    filtered_orbit       经过带通滤波的轴心轨迹（分频次提取）
    orbit_stats          轨迹统计量（外接椭圆、面积、偏心等）
    keyphasor_mark       键相器标记点计算（用于相位分析）
"""

import numpy as np
from typing import Tuple, Optional, Dict
from . import filters as _filters


def orbit_plot_data(x: np.ndarray, y: np.ndarray,
                    fs: float,
                    lp_cutoff: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算轴心轨迹数据。

    轴心轨迹由两个相互垂直方向（X、Y）的位移信号合成。
    通常来自两个成 90° 安装的电涡流传感器。

    Parameters
    ----------
    x, y      : 两路位移信号（µm），长度相等
    fs        : 采样频率 (Hz)
    lp_cutoff : 低通截止频率 (Hz)；不为 None 时先对两路信号滤波平滑

    Returns
    -------
    t  : np.ndarray，时间轴 (s)
    xf : np.ndarray，处理后的 X 位移
    yf : np.ndarray，处理后的 Y 位移
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    N = len(x)
    t = np.arange(N) / fs
    if lp_cutoff is not None:
        x = _filters.lowpass(x, lp_cutoff, fs)
        y = _filters.lowpass(y, lp_cutoff, fs)
    return t, x, y


def filtered_orbit(x: np.ndarray, y: np.ndarray,
                   fs: float,
                   rpm: float,
                   order: int = 1,
                   bw_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取特定阶次的轴心轨迹分量（同步滤波）。

    Parameters
    ----------
    x, y     : 两路位移信号
    fs       : 采样频率 (Hz)
    rpm      : 转速 (r/min)
    order    : 提取的旋转阶次（1X, 2X, ...），默认 1
    bw_ratio : 带通带宽相对于目标频率的比例，默认 0.1（即 ±5%）

    Returns
    -------
    xf, yf : 滤波后的 X、Y 位移
    """
    fn = rpm / 60.0 * order     # 目标频率 (Hz)
    bw = fn * bw_ratio
    low = max(fn - bw, 1.0)
    high = min(fn + bw, fs / 2 - 1)
    xf = _filters.bandpass(x, low, high, fs)
    yf = _filters.bandpass(y, low, high, fs)
    return xf, yf


def orbit_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    轴心轨迹统计特征。

    Parameters
    ----------
    x, y : 两路位移信号

    Returns
    -------
    dict，包含：
        x_rms, y_rms  : 均方根值
        x_peak, y_peak: 峰值
        total_vibration: 合成振幅 sqrt(x_rms² + y_rms²)
        eccentricity  : 直流偏置距离（轴心静态偏心量）
        orbit_area    : 用梯形公式估算的轨迹包围面积（µm²）
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_rms = float(np.sqrt(np.mean(x ** 2)))
    y_rms = float(np.sqrt(np.mean(y ** 2)))
    x_peak = float(np.max(np.abs(x)))
    y_peak = float(np.max(np.abs(y)))
    total = float(np.sqrt(x_rms ** 2 + y_rms ** 2))
    ecc = float(np.sqrt(np.mean(x) ** 2 + np.mean(y) ** 2))
    # Shoelace 公式估算面积
    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )
    return dict(x_rms=x_rms, y_rms=y_rms,
                x_peak=x_peak, y_peak=y_peak,
                total_vibration=total,
                eccentricity=ecc,
                orbit_area=float(area))


def keyphasor_mark(signal: np.ndarray, fs: float,
                   threshold: float = None,
                   edge: str = 'rising') -> np.ndarray:
    """
    从键相器信号中提取触发时刻（用于相位分析）。

    Parameters
    ----------
    signal    : 键相器脉冲信号
    fs        : 采样频率 (Hz)
    threshold : 触发电平；None 时取信号幅值范围的 50%
    edge      : 触发边沿 'rising'（上升）或 'falling'（下降）

    Returns
    -------
    times : np.ndarray，触发时刻 (s) 数组
    """
    s = np.asarray(signal, float)
    if threshold is None:
        threshold = (np.max(s) + np.min(s)) / 2.0
    above = s >= threshold
    if edge == 'rising':
        trigger = np.where(np.diff(above.astype(int)) == 1)[0]
    else:
        trigger = np.where(np.diff(above.astype(int)) == -1)[0]
    return trigger / fs
