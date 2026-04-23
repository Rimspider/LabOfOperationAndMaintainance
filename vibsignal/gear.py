"""
vibsignal.gear — 齿轮系统振动监测与故障诊断模块

提供以下功能：
    mesh_frequency         计算啮合频率
    tsa                    时域同步平均（TSA）
    residual_signal        残差信号（TSA 后去除啮合谐波）
    difference_signal      差分信号（TSA 后去除低阶多项式）
    gear_fault_indices     NA4、FM4 等齿轮故障指标
    mesh_spectrum          啮合频率幅值谱及边带分析
"""

import numpy as np
from typing import Optional, Tuple, Dict
from . import filters as _filters
from . import freq_domain as _fd


def mesh_frequency(rpm: float, n_teeth: int) -> float:
    """
    计算齿轮啮合频率。

    Parameters
    ----------
    rpm     : 转速 (r/min)
    n_teeth : 齿轮齿数

    Returns
    -------
    float，啮合频率 (Hz)
    """
    return rpm / 60.0 * n_teeth


def tsa(data: np.ndarray, fs: float,
        rpm: float,
        n_avg: int = None,
        keyphasor: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    时域同步平均（TSA, Time Synchronous Averaging）。

    将信号按旋转周期切割后叠加平均，提取同步于旋转的周期成分，
    抑制非同步噪声和其他轴的干扰。

    Parameters
    ----------
    data       : 振动信号
    fs         : 采样频率 (Hz)
    rpm        : 转速 (r/min)（稳态假设）
    n_avg      : 平均周期数；None 时取所有完整周期
    keyphasor  : 键相器触发时刻数组 (s)（若提供则优先使用）

    Returns
    -------
    t_avg  : np.ndarray，一个周期内的时间轴 (s)
    tsa_sig: np.ndarray，同步平均后的一个周期波形
    """
    fr = rpm / 60.0
    period_samples = int(round(fs / fr))
    N = len(data)

    if keyphasor is not None:
        starts = (keyphasor * fs).astype(int)
    else:
        n_periods = N // period_samples
        starts = np.arange(n_periods) * period_samples

    if n_avg is not None:
        starts = starts[:n_avg]

    # 截取有效帧
    valid = [s for s in starts if s + period_samples <= N]
    if len(valid) == 0:
        raise ValueError("有效周期数为 0，请检查转速或信号长度。")

    frames = np.array([data[s:s + period_samples] for s in valid])
    tsa_sig = np.mean(frames, axis=0)
    t_avg = np.arange(period_samples) / fs
    return t_avg, tsa_sig


def residual_signal(tsa_sig: np.ndarray, fs: float,
                    rpm: float, n_teeth: int,
                    n_mesh_harmonics: int = 10) -> np.ndarray:
    """
    残差信号 = TSA 信号 − 啮合频率谐波重建信号。

    去除啮合频率及其各次谐波（用 FFT 清零后 IFFT 重建），
    残余部分反映局部故障（断齿、点蚀等）。

    Parameters
    ----------
    tsa_sig          : TSA 后的一周期波形
    fs               : 采样频率 (Hz)
    rpm              : 转速 (r/min)
    n_teeth          : 齿数
    n_mesh_harmonics : 去除的啮合谐波次数，默认 10

    Returns
    -------
    residual : np.ndarray，残差信号
    """
    N = len(tsa_sig)
    fm = mesh_frequency(rpm, n_teeth)
    freq_res = fs / N
    fft_vals = np.fft.fft(tsa_sig)
    for k in range(1, n_mesh_harmonics + 1):
        target = fm * k
        idx = int(round(target / freq_res))
        if idx < N // 2:
            fft_vals[idx] = 0
            fft_vals[N - idx] = 0
    residual = np.real(np.fft.ifft(fft_vals))
    return residual


def difference_signal(tsa_sig: np.ndarray,
                      poly_order: int = 4) -> np.ndarray:
    """
    差分信号 = TSA 信号 − 多项式拟合趋势（去除低阶背景）。

    Parameters
    ----------
    tsa_sig    : TSA 后的一周期波形
    poly_order : 多项式阶数，默认 4

    Returns
    -------
    diff_sig : np.ndarray，差分信号
    """
    N = len(tsa_sig)
    x = np.arange(N)
    coeffs = np.polyfit(x, tsa_sig, poly_order)
    trend = np.polyval(coeffs, x)
    return tsa_sig - trend


def gear_fault_indices(tsa_sig: np.ndarray,
                       residual: np.ndarray) -> Dict[str, float]:
    """
    齿轮故障诊断指标。

    Parameters
    ----------
    tsa_sig  : TSA 同步平均信号（一周期）
    residual : 残差信号（residual_signal() 的输出）

    Returns
    -------
    dict，包含：
        FM4     : 残差信号的四阶矩指标（类似峭度，>4 为异常）
        NA4     : 归一化 FM4（FM4 除以方差的平方）
        RMS     : 残差均方根
        kurtosis: 残差峭度
    """
    from . import time_domain as td
    rms_res = td.rms(residual)
    var_res = np.var(residual)
    N = len(residual)
    mean_res = np.mean(residual)
    fm4 = np.mean((residual - mean_res) ** 4) / (var_res ** 2) if var_res > 0 else 0.0
    # NA4：用 TSA 信号的方差归一化
    var_tsa = np.var(tsa_sig)
    na4 = np.mean((residual - mean_res) ** 4) / (var_tsa ** 2) if var_tsa > 0 else 0.0
    kurt = td.kurtosis(residual)
    return dict(FM4=float(fm4), NA4=float(na4), RMS=float(rms_res), kurtosis=float(kurt))


def mesh_spectrum(data: np.ndarray, fs: float,
                  rpm: float, n_teeth: int,
                  n_mesh_harmonics: int = 5,
                  n_sidebands: int = 3,
                  window: str = 'hann') -> dict:
    """
    啮合频率幅值谱及边带分析。

    Parameters
    ----------
    data             : 振动信号
    fs               : 采样频率 (Hz)
    rpm              : 转速 (r/min)
    n_teeth          : 齿数
    n_mesh_harmonics : 分析的啮合谐波次数，默认 5
    n_sidebands      : 每个啮合谐波旁边的边带数，默认 3
    window           : 加窗类型，默认 'hann'

    Returns
    -------
    dict，包含：
        freq, ampl         : 幅值谱
        mesh_freq          : 啮合频率 (Hz)
        harmonics          : 各次啮合谐波幅值 list of (order, freq, ampl)
        sidebands          : 各次啮合谐波边带 dict{order: sideband_dict}
    """
    fm = mesh_frequency(rpm, n_teeth)
    fr = rpm / 60.0
    freq, ampl = _fd.amplitude_spectrum(data, fs, window=window)
    harmonics = _fd.harmonic_amplitudes(freq, ampl, fm, n_mesh_harmonics)
    sidebands = {}
    for k in range(1, n_mesh_harmonics + 1):
        sidebands[k] = _fd.sideband_amplitudes(
            freq, ampl, fm * k, fr, n_sidebands)
    return dict(freq=freq, ampl=ampl, mesh_freq=fm,
                harmonics=harmonics, sidebands=sidebands)
