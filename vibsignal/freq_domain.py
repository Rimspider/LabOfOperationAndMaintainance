"""
vibsignal.freq_domain — 频域分析模块

提供以下功能：
    amplitude_spectrum   幅值谱（单边）
    power_spectrum       功率谱密度（Welch 法）
    cepstrum             倒频谱
    order_spectrum       阶次谱（等角度重采样后的幅值谱）
    harmonic_amplitudes  谐波幅值提取
    sideband_amplitudes  边带幅值提取（齿轮诊断）
    dominant_frequencies 主频率提取（峰值搜索）
"""

import numpy as np
from scipy import signal as sp_signal
from typing import List, Tuple


def amplitude_spectrum(data: np.ndarray, fs: float,
                       window: str = 'hann',
                       one_sided: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算幅值谱（FFT）。

    Parameters
    ----------
    data      : 输入信号
    fs        : 采样频率 (Hz)
    window    : 加窗类型（'hann', 'hamming', 'blackman', 'rectangular'），默认 'hann'
    one_sided : 是否仅返回单边（正频率），默认 True

    Returns
    -------
    freq  : np.ndarray，频率轴 (Hz)
    ampl  : np.ndarray，幅值（与原信号单位相同）
    """
    x = np.asarray(data, float)
    N = len(x)
    if window == 'rectangular':
        win = np.ones(N)
    else:
        win = sp_signal.get_window(window, N)
    # 幅值校正系数
    acf = N / np.sum(win)
    xw = x * win
    fft_vals = np.fft.rfft(xw) if one_sided else np.fft.fft(xw)
    freq = np.fft.rfftfreq(N, d=1.0/fs) if one_sided else np.fft.fftfreq(N, d=1.0/fs)
    ampl = np.abs(fft_vals) / N * acf
    if one_sided:
        ampl[1:-1] *= 2          # 补偿折叠的负频部分
    return freq, ampl


def power_spectrum(data: np.ndarray, fs: float,
                   nperseg: int = None,
                   window: str = 'hann',
                   overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    功率谱密度（Welch 平均周期图法）。

    Parameters
    ----------
    data     : 输入信号
    fs       : 采样频率 (Hz)
    nperseg  : 每段长度；None 时自动取 len(data)//8（至少 256）
    window   : 窗函数类型，默认 'hann'
    overlap  : 重叠比例（0~1），默认 0.5

    Returns
    -------
    freq : np.ndarray，频率轴 (Hz)
    psd  : np.ndarray，功率谱密度 (单位²/Hz)
    """
    x = np.asarray(data, float)
    if nperseg is None:
        # Default segment length: len(data)//8, but clamp to [10, 1500]
        nperseg = len(x) // 8 if len(x) >= 8 else len(x)
        nperseg = int(max(10, min(1500, nperseg)))
    noverlap = int(nperseg * overlap)
    freq, psd = sp_signal.welch(x, fs=fs, window=window,
                                nperseg=nperseg, noverlap=noverlap)
    return freq, psd


def cepstrum(data: np.ndarray, fs: float,
             window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """
    实倒频谱分析（用于识别等间隔谐波族，如齿轮故障的边带族）。

    C(q) = |IFFT{log|FFT{x}|}|

    Parameters
    ----------
    data   : 输入信号
    fs     : 采样频率 (Hz)
    window : 加窗类型，默认 'hann'

    Returns
    -------
    quefrency : np.ndarray，倒频率轴（单位：s，取单边正倒频率）
    ceps      : np.ndarray，倒频谱幅值
    """
    x = np.asarray(data, float)
    N = len(x)
    win = sp_signal.get_window(window, N)
    xw = x * win
    # FFT → 取对数 → IFFT → 取实部绝对值
    spectrum = np.fft.fft(xw)
    log_spec = np.log(np.abs(spectrum) + 1e-12)
    ceps_full = np.real(np.fft.ifft(log_spec))
    quefrency = np.arange(N) / fs
    # 只返回单边正倒频率（去除直流 0）
    half = N // 2
    return quefrency[1:half], np.abs(ceps_full[1:half])


def dominant_frequencies(freq: np.ndarray, ampl: np.ndarray,
                         n: int = 10,
                         min_distance_hz: float = None) -> List[Tuple[float, float]]:
    """
    从幅值谱中提取 n 个主频率（峰值搜索）。

    Parameters
    ----------
    freq             : 频率轴
    ampl             : 幅值轴
    n                : 提取峰值数量，默认 10
    min_distance_hz  : 峰间最小距离 (Hz)；None 时自动设为 freq 分辨率的 5 倍

    Returns
    -------
    list of (freq_hz, amplitude) 按幅值从大到小排序
    """
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0
    if min_distance_hz is None:
        min_distance_hz = df * 5
    min_distance = max(1, int(min_distance_hz / df))
    peaks, _ = sp_signal.find_peaks(ampl, distance=min_distance)
    if len(peaks) == 0:
        return []
    sorted_idx = peaks[np.argsort(ampl[peaks])[::-1]]
    top = sorted_idx[:n]
    return [(float(freq[i]), float(ampl[i])) for i in top]


def harmonic_amplitudes(freq: np.ndarray, ampl: np.ndarray,
                        fundamental: float,
                        n_harmonics: int = 5,
                        tol_hz: float = None) -> List[Tuple[int, float, float]]:
    """
    提取基频及其各次谐波的幅值。

    Parameters
    ----------
    freq        : 频率轴 (Hz)
    ampl        : 幅值轴
    fundamental : 基频 (Hz)
    n_harmonics : 谐波次数（含基频），默认 5
    tol_hz      : 搜索容差 (Hz)；默认为基频的 5%

    Returns
    -------
    list of (order, freq_hz, amplitude)
    """
    if tol_hz is None:
        tol_hz = fundamental * 0.05
    results = []
    for k in range(1, n_harmonics + 1):
        target = fundamental * k
        mask = np.abs(freq - target) <= tol_hz
        if np.any(mask):
            idx = np.argmax(ampl * mask)
            results.append((k, float(freq[idx]), float(ampl[idx])))
        else:
            results.append((k, target, 0.0))
    return results


def sideband_amplitudes(freq: np.ndarray, ampl: np.ndarray,
                        center_freq: float,
                        modulation_freq: float,
                        n_sidebands: int = 3,
                        tol_hz: float = None) -> dict:
    """
    提取齿轮啮合频率边带（用于齿轮故障诊断）。

    Parameters
    ----------
    freq            : 频率轴 (Hz)
    ampl            : 幅值轴
    center_freq     : 中心频率（通常为啮合频率或其谐波）(Hz)
    modulation_freq : 调制频率（通常为转频）(Hz)
    n_sidebands     : 单边带数量，默认 3
    tol_hz          : 搜索容差 (Hz)

    Returns
    -------
    dict，键为 0（载波）、±1、±2 ... 的边带阶次，值为 (freq_hz, amplitude)
    """
    if tol_hz is None:
        tol_hz = modulation_freq * 0.1

    def _pick(target):
        mask = np.abs(freq - target) <= tol_hz
        if np.any(mask):
            idx = np.argmax(ampl * mask)
            return float(freq[idx]), float(ampl[idx])
        return target, 0.0

    result = {}
    result[0] = _pick(center_freq)
    for k in range(1, n_sidebands + 1):
        result[+k] = _pick(center_freq + k * modulation_freq)
        result[-k] = _pick(center_freq - k * modulation_freq)
    return result
