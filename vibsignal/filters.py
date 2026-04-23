"""
vibsignal.filters — 滤波器模块

提供以下滤波器：
    lowpass       低通滤波
    highpass      高通滤波
    bandpass      带通滤波
    bandstop      带阻滤波
    envelope      包络检波（Hilbert 解调）
    median_filter 中值滤波（去毛刺）
    moving_avg    移动平均（趋势提取）
"""

import numpy as np
from scipy import signal as sp_signal


def _butter_filter(data: np.ndarray, cutoff, fs: float,
                   btype: str, order: int = 4) -> np.ndarray:
    """内部通用 Butterworth 滤波器。"""
    nyq = 0.5 * fs
    if np.isscalar(cutoff):
        norm = cutoff / nyq
    else:
        norm = [c / nyq for c in cutoff]
    b, a = sp_signal.butter(order, norm, btype=btype)
    return sp_signal.filtfilt(b, a, data)


def lowpass(data: np.ndarray, cutoff: float, fs: float,
            order: int = 4) -> np.ndarray:
    """
    低通滤波器（Butterworth 零相位）。

    Parameters
    ----------
    data   : 输入信号
    cutoff : 截止频率 (Hz)
    fs     : 采样频率 (Hz)
    order  : 滤波器阶数，默认 4

    Returns
    -------
    np.ndarray  滤波后信号
    """
    return _butter_filter(data, cutoff, fs, 'low', order)


def highpass(data: np.ndarray, cutoff: float, fs: float,
             order: int = 4) -> np.ndarray:
    """
    高通滤波器（Butterworth 零相位）。

    Parameters
    ----------
    data   : 输入信号
    cutoff : 截止频率 (Hz)
    fs     : 采样频率 (Hz)
    order  : 滤波器阶数，默认 4
    """
    return _butter_filter(data, cutoff, fs, 'high', order)


def bandpass(data: np.ndarray, low: float, high: float,
             fs: float, order: int = 4) -> np.ndarray:
    """
    带通滤波器（Butterworth 零相位）。

    Parameters
    ----------
    data  : 输入信号
    low   : 低截止频率 (Hz)
    high  : 高截止频率 (Hz)
    fs    : 采样频率 (Hz)
    order : 滤波器阶数，默认 4
    """
    return _butter_filter(data, [low, high], fs, 'bandpass', order)


def bandstop(data: np.ndarray, low: float, high: float,
             fs: float, order: int = 4) -> np.ndarray:
    """
    带阻（陷波）滤波器（Butterworth 零相位）。

    Parameters
    ----------
    data  : 输入信号
    low   : 阻带低频 (Hz)
    high  : 阻带高频 (Hz)
    fs    : 采样频率 (Hz)
    order : 滤波器阶数，默认 4
    """
    return _butter_filter(data, [low, high], fs, 'bandstop', order)


def envelope(data: np.ndarray, fs: float,
             lp_cutoff: float = None) -> np.ndarray:
    """
    包络检波（Hilbert 变换解调）。

    先对信号取 Hilbert 变换，再求解析信号的模（瞬时幅值），
    可选对结果做低通滤波平滑。

    Parameters
    ----------
    data      : 输入信号（建议先经过带通滤波去除低频和高频噪声）
    fs        : 采样频率 (Hz)
    lp_cutoff : 低通截止频率 (Hz)；为 None 时不做平滑

    Returns
    -------
    np.ndarray  包络信号
    """
    analytic = sp_signal.hilbert(data)
    env = np.abs(analytic)
    if lp_cutoff is not None:
        env = lowpass(env, lp_cutoff, fs)
    return env


def median_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    中值滤波，用于去除脉冲噪声（毛刺）。

    Parameters
    ----------
    data        : 输入信号
    kernel_size : 滑动窗口长度（奇数），默认 5
    """
    return sp_signal.medfilt(data, kernel_size=kernel_size)


def moving_avg(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    移动平均滤波，用于提取信号趋势。

    Parameters
    ----------
    data   : 输入信号
    window : 窗口长度，默认 10

    Returns
    -------
    np.ndarray  与输入等长的移动平均信号（边界用 'same' 卷积模式填充）
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')
