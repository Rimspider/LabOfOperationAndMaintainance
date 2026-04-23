"""
vibsignal.bearing — 滚动轴承故障诊断模块

提供以下功能：
    fault_frequencies   计算四类故障特征频率
    envelope_spectrum   包络谱（用于轴承故障频率识别）
    diagnosis           自动故障诊断（匹配特征频率）
    bearing_health      综合健康评分
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from . import filters as _filters
from . import freq_domain as _fd


def fault_frequencies(rpm: float, n_balls: int,
                      d_ball: float, D_pitch: float,
                      contact_angle_deg: float = 0.0) -> Dict[str, float]:
    """
    计算滚动轴承四类故障特征频率。

    参数说明（遵循国标 GB/T 6391）：
    ─────────────────────────────────────────
    外圈故障频率 BPFO = (n/2) * fr * (1 - Bd/Pd * cos α)
    内圈故障频率 BPFI = (n/2) * fr * (1 + Bd/Pd * cos α)
    滚动体故障频率 BSF = (Pd/(2*Bd)) * fr * (1 - (Bd/Pd * cos α)²)
    保持架故障频率 FTF = (fr/2) * (1 - Bd/Pd * cos α)
    ─────────────────────────────────────────

    Parameters
    ----------
    rpm              : 转速 (r/min)
    n_balls          : 滚动体数量
    d_ball           : 滚动体直径 (mm 或任意一致单位)
    D_pitch          : 节径（滚动体中心圆直径）(mm)
    contact_angle_deg: 接触角 (°)，默认 0°

    Returns
    -------
    dict，包含：
        fr   : 转频 (Hz)
        BPFO : 外圈故障频率 (Hz)
        BPFI : 内圈故障频率 (Hz)
        BSF  : 滚动体故障频率 (Hz)
        FTF  : 保持架故障频率 (Hz)
    """
    fr = rpm / 60.0
    alpha = np.deg2rad(contact_angle_deg)
    ratio = (d_ball / D_pitch) * np.cos(alpha)

    BPFO = (n_balls / 2) * fr * (1 - ratio)
    BPFI = (n_balls / 2) * fr * (1 + ratio)
    BSF  = (D_pitch / (2 * d_ball)) * fr * (1 - ratio ** 2)
    FTF  = (fr / 2) * (1 - ratio)

    return dict(fr=fr, BPFO=BPFO, BPFI=BPFI, BSF=BSF, FTF=FTF)


def envelope_spectrum(data: np.ndarray, fs: float,
                      bp_low: float, bp_high: float,
                      window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """
    包络谱分析（Hilbert 解调 + FFT）。

    步骤：
      1. 带通滤波（聚焦共振频带）
      2. Hilbert 变换 → 包络
      3. 去直流（减均值）
      4. FFT → 包络幅值谱

    Parameters
    ----------
    data     : 原始振动信号（加速度）
    fs       : 采样频率 (Hz)
    bp_low   : 带通下限频率 (Hz)（共振频带下限，通常 2~10 kHz）
    bp_high  : 带通上限频率 (Hz)（共振频带上限）
    window   : 加窗类型，默认 'hann'

    Returns
    -------
    freq : np.ndarray，频率轴 (Hz)
    ampl : np.ndarray，包络幅值谱
    """
    x = _filters.bandpass(data, bp_low, bp_high, fs)
    env = _filters.envelope(x, fs)
    env -= np.mean(env)
    freq, ampl = _fd.amplitude_spectrum(env, fs, window=window)
    return freq, ampl


def diagnosis(freq: np.ndarray, ampl: np.ndarray,
              fault_freqs: Dict[str, float],
              n_harmonics: int = 3,
              tol_ratio: float = 0.05,
              threshold_ratio: float = 0.1) -> Dict[str, dict]:
    """
    根据包络谱自动识别轴承故障类型。

    将峰值谱与各故障特征频率（及其谐波）进行匹配。

    Parameters
    ----------
    freq            : 包络谱频率轴
    ampl            : 包络谱幅值
    fault_freqs     : fault_frequencies() 返回的字典
    n_harmonics     : 搜索谐波次数，默认 3
    tol_ratio       : 频率匹配容差（相对于目标频率），默认 5%
    threshold_ratio : 幅值阈值（相对于最大幅值），默认 10%

    Returns
    -------
    dict，每个故障类型（'BPFO','BPFI','BSF','FTF'）对应：
        matched_harmonics : 匹配到的谐波次数列表
        peak_amplitudes   : 对应幅值
        score             : 匹配得分（0~1）
        detected          : 是否超阈值判定为故障
    """
    max_ampl = np.max(ampl)
    results = {}
    for fault_type in ('BPFO', 'BPFI', 'BSF', 'FTF'):
        f0 = fault_freqs.get(fault_type, 0)
        if f0 <= 0:
            continue
        matched_orders, matched_ampls = [], []
        for k in range(1, n_harmonics + 1):
            target = f0 * k
            tol = target * tol_ratio
            mask = np.abs(freq - target) <= tol
            if np.any(mask):
                idx = np.argmax(ampl * mask)
                if ampl[idx] >= max_ampl * threshold_ratio:
                    matched_orders.append(k)
                    matched_ampls.append(float(ampl[idx]))
        score = len(matched_orders) / n_harmonics
        results[fault_type] = dict(
            matched_harmonics=matched_orders,
            peak_amplitudes=matched_ampls,
            score=score,
            detected=score >= 1 / n_harmonics
        )
    return results


def bearing_health(data: np.ndarray) -> Dict[str, float]:
    """
    基于时域特征的轴承健康指标（无需已知故障频率）。

    Returns
    -------
    dict，包含：
        kurtosis      峭度（>3 时可能有早期故障）
        crest_factor  波峰因数（>6 异常）
        rms           均方根（趋势监测）
        impulse_factor脉冲因数
    """
    from . import time_domain as td
    s = td.stats(data)
    return dict(
        kurtosis=s['kurtosis'],
        crest_factor=s['crest_factor'],
        rms=s['rms'],
        impulse_factor=s['impulse_factor']
    )
