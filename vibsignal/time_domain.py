"""
vibsignal.time_domain — 时域分析模块

提供以下功能：
    stats           统计特征（均值、均方根、峰值、波峰因数、峭度等）
    autocorrelation 自相关分析
    crosscorrelation 互相关分析
    crest_factor    波峰因数
    kurtosis        峭度
    impulse_factor  脉冲因数
    margin_factor   裕度因数
    shape_factor    波形因数
    summary         打印全部特征
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis as _scipy_kurtosis
from typing import Dict


def stats(data: np.ndarray) -> Dict[str, float]:
    """
    计算信号的全套时域统计特征。

    Parameters
    ----------
    data : 输入振动信号（一维数组）

    Returns
    -------
    dict，包含以下键：
        mean          均值
        std           标准差
        rms           均方根值（有效值）
        peak          峰值（最大绝对值）
        peak_to_peak  峰峰值
        crest_factor  波峰因数 = peak / rms
        kurtosis      峭度（Fisher 定义，正态分布 ≈ 0）
        skewness      偏度
        shape_factor  波形因数 = rms / |mean|（|mean|≈0 时为 inf）
        impulse_factor脉冲因数 = peak / (mean of |x|)
        margin_factor 裕度因数 = peak / (mean of sqrt|x|)²
    """
    x = np.asarray(data, dtype=float)
    mean = np.mean(x)
    std  = np.std(x, ddof=1)
    rms  = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    p2p  = np.max(x) - np.min(x)
    cf   = peak / rms if rms > 0 else np.inf
    kurt = _scipy_kurtosis(x, fisher=True, bias=False)   # Fisher: 正态=0
    skew = float(np.mean(((x - mean) / std) ** 3)) if std > 0 else 0.0
    abs_mean = np.mean(np.abs(x))
    sf   = rms / abs_mean if abs_mean > 0 else np.inf
    impf = peak / abs_mean if abs_mean > 0 else np.inf
    sqrt_mean = np.mean(np.sqrt(np.abs(x))) ** 2
    mf   = peak / sqrt_mean if sqrt_mean > 0 else np.inf
    return dict(mean=mean, std=std, rms=rms, peak=peak,
                peak_to_peak=p2p, crest_factor=cf, kurtosis=kurt,
                skewness=skew, shape_factor=sf,
                impulse_factor=impf, margin_factor=mf)


def rms(data: np.ndarray) -> float:
    """均方根值。"""
    return float(np.sqrt(np.mean(np.asarray(data, float) ** 2)))


def crest_factor(data: np.ndarray) -> float:
    """波峰因数 = 峰值 / 均方根值。"""
    x = np.asarray(data, float)
    r = rms(x)
    return float(np.max(np.abs(x)) / r) if r > 0 else np.inf


def kurtosis(data: np.ndarray) -> float:
    """
    峭度（Fisher 定义，正态分布约为 0）。
    轴承早期故障时峭度显著升高（通常 > 3 为异常参考值）。
    """
    return float(_scipy_kurtosis(data, fisher=True, bias=False))


def impulse_factor(data: np.ndarray) -> float:
    """脉冲因数 = 峰值 / 绝对均值。"""
    x = np.asarray(data, float)
    am = np.mean(np.abs(x))
    return float(np.max(np.abs(x)) / am) if am > 0 else np.inf


def margin_factor(data: np.ndarray) -> float:
    """裕度因数 = 峰值 / 方根幅值²。"""
    x = np.asarray(data, float)
    sm = np.mean(np.sqrt(np.abs(x))) ** 2
    return float(np.max(np.abs(x)) / sm) if sm > 0 else np.inf


def shape_factor(data: np.ndarray) -> float:
    """波形因数 = 均方根 / 绝对均值。"""
    x = np.asarray(data, float)
    am = np.mean(np.abs(x))
    return float(rms(x) / am) if am > 0 else np.inf


def autocorrelation(data: np.ndarray,
                    max_lag: int = None,
                    normalize: bool = True) -> tuple:
    """
    自相关分析。

    Parameters
    ----------
    data      : 输入信号
    max_lag   : 最大延迟点数；默认为信号长度 - 1
    normalize : 是否归一化（除以零延迟值），默认 True

    Returns
    -------
    lags  : np.ndarray，延迟点数数组（含负延迟）
    Rxx   : np.ndarray，自相关函数值
    """
    x = np.asarray(data, float) - np.mean(data)
    N = len(x)
    if max_lag is None:
        max_lag = N - 1
    max_lag = min(max_lag, N - 1)

    Rxx_full = np.correlate(x, x, mode='full')
    center = N - 1
    Rxx = Rxx_full[center - max_lag: center + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)

    if normalize and Rxx_full[center] != 0:
        Rxx = Rxx / Rxx_full[center]
    return lags, Rxx


def crosscorrelation(x: np.ndarray, y: np.ndarray,
                     max_lag: int = None,
                     normalize: bool = True) -> tuple:
    """
    互相关分析（用于分析两路信号的时延关系）。

    Parameters
    ----------
    x, y      : 两路输入信号（等长）
    max_lag   : 最大延迟点数；默认为信号长度 - 1
    normalize : 是否归一化，默认 True

    Returns
    -------
    lags  : np.ndarray
    Rxy   : np.ndarray，互相关函数值
    """
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    N = max(len(x), len(y))
    if max_lag is None:
        max_lag = N - 1
    max_lag = min(max_lag, N - 1)

    Rxy_full = np.correlate(x, y, mode='full')
    center = len(x) - 1
    Rxy = Rxy_full[center - max_lag: center + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)

    if normalize:
        norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        if norm > 0:
            Rxy = Rxy / norm
    return lags, Rxy


def summary(data: np.ndarray) -> None:
    """打印全部时域特征。"""
    s = stats(data)
    print("=" * 40)
    print("  时域特征统计")
    print("=" * 40)
    labels = {
        'mean':           '均值          ',
        'std':            '标准差        ',
        'rms':            '均方根值      ',
        'peak':           '峰值          ',
        'peak_to_peak':   '峰峰值        ',
        'crest_factor':   '波峰因数      ',
        'kurtosis':       '峭度          ',
        'skewness':       '偏度          ',
        'shape_factor':   '波形因数      ',
        'impulse_factor': '脉冲因数      ',
        'margin_factor':  '裕度因数      ',
    }
    for k, label in labels.items():
        print(f"  {label}: {s[k]:.6g}")
    print("=" * 40)
