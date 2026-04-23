"""
vibsignal.time_freq — 时频分析模块

提供以下功能：
    stft               短时傅里叶变换（STFT）
    cwt                连续小波变换（CWT）
    hilbert_envelope   Hilbert 变换 → 瞬时属性（幅值、频率、相位）
    wvd                Wigner-Ville 分布（离散近似）
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple


def stft(data: np.ndarray, fs: float,
         window_size: float = 0.02,
         overlap: float = 0.75,
         window: str = 'hann',
         nfft: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    短时傅里叶变换（STFT）。

    Parameters
    ----------
    data        : 输入信号
    fs          : 采样频率 (Hz)
    window_size : 窗口时长 (s)，默认 0.02 s
    overlap     : 重叠比例（0~1），默认 0.75
    window      : 窗函数类型，默认 'hann'
    nfft        : FFT 点数；None 时取窗口长度的下一个 2 的幂次

    Returns
    -------
    t    : np.ndarray，时间轴 (s)
    freq : np.ndarray，频率轴 (Hz)
    Zxx  : np.ndarray，幅值矩阵（shape: [freq, time]）
    """
    nperseg = int(window_size * fs)
    noverlap = int(nperseg * overlap)
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(nperseg)))

    freq, t, Zxx = sp_signal.stft(
        data, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    return t, freq, np.abs(Zxx)


def cwt(data: np.ndarray, fs: float,
        wavelet: str = 'morl',
        freqs: np.ndarray = None,
        min_freq: float = 1.0,
        max_freq: float = None,
        n_freqs: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    连续小波变换（CWT）。使用 scipy.signal.morlet2 实现 Morlet 小波。

    Parameters
    ----------
    data      : 输入信号
    fs        : 采样频率 (Hz)
    wavelet   : 小波类型，目前支持 'morl'（Morlet）
    freqs     : 自定义频率轴；为 None 时用 min_freq~max_freq 等对数间隔
    min_freq  : 最低分析频率 (Hz)，默认 1 Hz
    max_freq  : 最高分析频率 (Hz)；None 时取 fs/2
    n_freqs   : 频率分档数，默认 64

    Returns
    -------
    t      : np.ndarray，时间轴 (s)
    freqs  : np.ndarray，频率轴 (Hz)
    coefs  : np.ndarray，小波系数幅值（shape: [freq, time]）
    """
    x = np.asarray(data, float)
    N = len(x)
    t = np.arange(N) / fs

    if max_freq is None:
        max_freq = fs / 2.0
    if freqs is None:
        freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freqs)

    # 使用 Morlet 小波，w0=6 是标准选择（保证频率分辨率和时间分辨率的平衡）
    w0 = 6.0
    coefs = np.zeros((len(freqs), N), dtype=complex)
    for i, f in enumerate(freqs):
        # 尺度 a = w0*fs / (2*pi*f)
        widths = w0 * fs / (2 * np.pi * f)
        c = sp_signal.cwt(x, sp_signal.morlet2, [widths], w=w0)
        coefs[i, :] = c[0]

    return t, freqs, np.abs(coefs)


def hilbert_envelope(data: np.ndarray, fs: float) -> dict:
    """
    Hilbert 变换 → 解析信号 → 瞬时属性。

    Parameters
    ----------
    data : 输入信号（建议已经过带通滤波）
    fs   : 采样频率 (Hz)

    Returns
    -------
    dict，包含：
        t                 : 时间轴 (s)
        envelope          : 瞬时幅值（包络）
        instantaneous_phase: 瞬时相位 (rad)
        instantaneous_freq : 瞬时频率 (Hz)
    """
    x = np.asarray(data, float)
    N = len(x)
    t = np.arange(N) / fs
    analytic = sp_signal.hilbert(x)
    env = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi) * fs
    inst_freq = np.append(inst_freq, inst_freq[-1])   # 补齐长度
    return dict(t=t, envelope=env,
                instantaneous_phase=phase,
                instantaneous_freq=inst_freq)


def wvd(data: np.ndarray, fs: float,
        n_time: int = 128,
        n_freq: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    伪 Wigner-Ville 分布（PWVD）——带 Hann 窗的离散近似。

    适用于分析瞬时频率变化（如启停车过渡过程）。
    注：计算量较大，建议对信号降采样后使用。

    Parameters
    ----------
    data   : 输入信号
    fs     : 采样频率 (Hz)
    n_time : 时间抽样点数，默认 128
    n_freq : 频率分辨率（NFFT），默认 128

    Returns
    -------
    t    : np.ndarray，时间轴 (s)
    freq : np.ndarray，频率轴 (Hz)（单边）
    WVD  : np.ndarray，分布矩阵
    """
    x = np.asarray(data, float)
    N = len(x)
    # 解析信号
    z = sp_signal.hilbert(x)
    time_indices = np.linspace(0, N - 1, n_time, dtype=int)
    t = time_indices / fs
    freq = np.fft.rfftfreq(n_freq, d=1.0 / fs)

    half_win = n_freq // 2
    win = np.hanning(2 * half_win + 1)
    WVD = np.zeros((len(freq), n_time))

    for ti, n in enumerate(time_indices):
        kernel = np.zeros(n_freq, dtype=complex)
        for k in range(-half_win, half_win + 1):
            n1 = n + k
            n2 = n - k
            if 0 <= n1 < N and 0 <= n2 < N:
                kernel[k % n_freq] += win[k + half_win] * z[n1] * np.conj(z[n2])
        col = np.fft.rfft(kernel)
        WVD[:, ti] = np.real(col)

    return t, freq, WVD
