"""
vibsignal.viz — 可视化模块

所有函数均返回 matplotlib Figure 对象，方便保存或嵌入 Notebook。

提供以下绘图函数：
    time_waveform        时域波形图
    amplitude_spectrum   幅值谱图
    power_spectrum       功率谱密度图
    spectrogram          STFT 时频谱图（语谱图）
    scalogram            CWT 时频图
    orbit                轴心轨迹图
    correlation          自相关 / 互相关图
    cepstrum             倒频谱图
    envelope_spectrum    包络谱图（含故障频率标注）
    gear_mesh_spectrum   齿轮啮合频率边带图
    dashboard            多子图综合仪表盘（时域 + 频域 + 时频 + 统计）
    compare_signals      多信号对比图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')           # 非交互式后端，适合脚本与服务器
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib import font_manager as fm
from typing import Optional, Dict, List, Tuple
from . import freq_domain as _fd

# ── 全局样式配置 ──────────────────────────────────────────────────────────────
_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
           '#00BCD4', '#FFEB3B', '#795548']

def _style():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'figure.dpi':       120,
    })

_style()


def _ensure_chinese_font():
    """Try to set a Chinese-capable font if available on the system.

    This sets `plt.rcParams['font.family']` to the first matching candidate.
    """
    if getattr(_ensure_chinese_font, '_done', False):
        return
    candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Source Han Sans CN', 'PingFang SC'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        # match if any available font contains candidate substring
        for a in available:
            if name.lower() in a.lower():
                # set both family and sans-serif list so ticks/labels use it
                plt.rcParams['font.sans-serif'] = [a]
                plt.rcParams['font.family'] = 'sans-serif'
                # avoid missing minus sign glyph problems
                plt.rcParams['axes.unicode_minus'] = False
                _ensure_chinese_font._done = True
                return
    # leave default if none found


# ── 基础工具 ──────────────────────────────────────────────────────────────────

def _make_fig(figsize=(10, 4), title=None):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    return fig, ax


# ── 绘图函数 ──────────────────────────────────────────────────────────────────

def time_waveform(data: np.ndarray, fs: float,
                  label: str = 'Vibration',
                  unit: str = 'm/s²',
                  title: str = 'Time Waveform',
                  show_stats: bool = True) -> plt.Figure:
    """
    绘制时域波形图。

    Parameters
    ----------
    data       : 振动信号
    fs         : 采样频率 (Hz)
    label      : 图例名称
    unit       : Y 轴单位
    title      : 图标题
    show_stats : 是否在标题中显示 RMS/峰值/峭度

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_chinese_font()
    data_slice = slice(10, 3000)
    data = data[data_slice]
    t = np.arange(len(data)) / fs
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, data, color=_COLORS[0], lw=0.8, label=label)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(f'Amplitude', fontsize=11)

    if show_stats:
        from . import time_domain as td
        s = td.stats(data)
        sub = (f"RMS={s['rms']:.4g} | Peak={s['peak']:.4g} | "
               f"Kurtosis={s['kurtosis']:.4g} | Crest Factor={s['crest_factor']:.4g}")
        ax.set_title(f'{title}\n{sub}', fontsize=11)
    else:
        ax.set_title(title, fontsize=11)

    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    return fig


def amplitude_spectrum(freq: np.ndarray, ampl: np.ndarray,
                       title: str = 'Amplitude Spectrum',
                       unit: str = 'm/s²',
                       xmax: float = None,
                       mark_freqs: Dict[str, float] = None,
                       log_y: bool = False,
                       top_n: int = 10) -> plt.Figure:
    """
    绘制幅值谱。

    Parameters
    ----------
    freq        : 频率轴 (Hz)
    ampl        : 幅值轴
    title       : 图标题
    unit        : 幅值单位
    xmax        : X 轴上限 (Hz)；None 时自动
    mark_freqs  : 需要标注的特征频率，如 {'fr': 25.0, 'BPFO': 105.0}
    log_y       : Y 轴是否取对数

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freq, ampl, color=_COLORS[0], lw=0.8)
    if log_y:
        ax.set_yscale('log')
    # default x-axis frequency limit to 1000 Hz
    if xmax is None:
        xmax = 1000.0
    ax.set_xlim(0, xmax)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel(f'Amplitude', fontsize=11)
    ax.set_title(title, fontsize=11)

    if mark_freqs:
        colors = _COLORS[1:]
        for i, (name, f0) in enumerate(mark_freqs.items()):
            c = colors[i % len(colors)]
            ax.axvline(f0, color=c, ls='--', lw=1.0, alpha=0.8)
            ax.text(f0, ax.get_ylim()[1] * 0.92, name,
                    color=c, fontsize=8, ha='center')

    # 标注前 N 个峰值及其代表的功率（幅值的平方）
    if top_n and top_n > 0:
        try:
            # 仅在可见的 x 轴频率范围内搜索峰值
            visible_mask = (freq >= 0) & (freq <= xmax) if xmax is not None else np.ones_like(freq, dtype=bool)
            if np.any(visible_mask):
                f_vis = freq[visible_mask]
                a_vis = ampl[visible_mask]
                peaks = _fd.dominant_frequencies(f_vis, a_vis, n=top_n)
                for i, (f0, a0) in enumerate(peaks):
                    p0 = a0 * a0
                    ax.scatter([f0], [a0], color=_COLORS[(i+1) % len(_COLORS)], zorder=6)
                    ax.text(f0, a0 * 1.05, f"{f0:.2f} Hz",
                            fontsize=8, ha='center', color=_COLORS[(i+1) % len(_COLORS)])
        except Exception:
            pass

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    return fig


def power_spectrum(freq: np.ndarray, psd: np.ndarray,
                   title: str = 'Power Spectral Density (Welch)',
                   unit: str = 'm/s²',
                   xmax: float = None,
                   log_y: bool = True) -> plt.Figure:
    """Plot power spectral density (Welch)."""
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(freq, psd, color=_COLORS[0], lw=0.9) if log_y else ax.plot(freq, psd, color=_COLORS[0], lw=0.9)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel(f'PSD ', fontsize=11)
    ax.set_title(title, fontsize=11)
    # default x-axis frequency limit to 1000 Hz
    if xmax is None:
        xmax = 300.0
    ax.set_xlim(0, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    return fig


def spectrogram(t: np.ndarray, freq: np.ndarray, Zxx: np.ndarray,
                title: str = 'STFT Spectrogram',
                fmax: float = 1000.0,
                cmap: str = 'jet',
                log_z: bool = True) -> plt.Figure:
    """
    绘制 STFT 时频谱（语谱图）。

    Parameters
    ----------
    t, freq, Zxx : stft() 的返回值
    fmax         : 显示频率上限 (Hz)
    cmap         : 色图，默认 'jet'
    log_z        : 是否对幅值取 dB，默认 True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    Z = 20 * np.log10(Zxx + 1e-12) if log_z else Zxx
    freq_mask = (freq <= fmax) if fmax is not None else np.ones(len(freq), bool)

    im = ax.pcolormesh(t, freq[freq_mask], Z[freq_mask, :],
                       shading='gouraud', cmap=cmap)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('dB' if log_z else '幅值', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_xlim(0, fmax if fmax is not None else 300.0)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def scalogram(t: np.ndarray, freqs: np.ndarray, coefs: np.ndarray,
              title: str = 'CWT Scalogram',
              fmax: float = 1000.0,
              cmap: str = 'jet') -> plt.Figure:
    """绘制 CWT 时频图（鱼骨图）。"""
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    freq_mask = (freqs <= fmax) if fmax is not None else np.ones(len(freqs), bool)
    im = ax.pcolormesh(t, freqs[freq_mask], coefs[freq_mask, :],
                       shading='gouraud', cmap=cmap)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('系数幅值', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_xlim(0, fmax if fmax is not None else 300.0)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def orbit(x: np.ndarray, y: np.ndarray,
          x_label: str = 'X displacement (µm)',
          y_label: str = 'Y displacement (µm)',
          title: str = 'Orbit',
          color_by_time: bool = True) -> plt.Figure:
    """
    绘制轴心轨迹图。

    Parameters
    ----------
    x, y          : 两路位移信号
    x_label, y_label: 坐标轴标签
    title         : 图标题
    color_by_time : 是否按时间着色（Jet 渐变），默认 True

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(6, 6))
    if color_by_time:
        N = len(x)
        cmap = plt.get_cmap('jet')
        for i in range(N - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=cmap(i / N), lw=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                       norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Time (normalized)', shrink=0.8)
    else:
        ax.plot(x, y, color=_COLORS[0], lw=0.8)

    # 标记起始点
    ax.scatter(x[0], y[0], s=60, color='green', zorder=5, label='Start')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def correlation(lags: np.ndarray, corr: np.ndarray,
                title: str = 'Autocorrelation',
                fs: float = None) -> plt.Figure:
    """
    绘制自相关或互相关图。

    Parameters
    ----------
    lags  : 延迟轴（样本点数或 s）
    corr  : 相关函数值
    title : 图标题
    fs    : 若不为 None，则将 lags 转换为秒

    Returns
    -------
    matplotlib.figure.Figure
    """
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4))
    x_axis = lags / fs if fs is not None else lags
    xlabel = 'Delay (s)' if fs is not None else 'Delay (samples)'
    ax.plot(x_axis, corr, color=_COLORS[0], lw=0.8)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('相关系数', fontsize=11)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def cepstrum(quefrency: np.ndarray, ceps: np.ndarray,
             title: str = 'Cepstrum',
             qmax: float = None,
             top_n: int = 0) -> plt.Figure:
    """绘制倒频谱图。"""
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(quefrency, ceps, color=_COLORS[0], lw=0.8)
    if qmax:
        ax.set_xlim(0, qmax)
    ax.set_xlabel('Quefrency (s)', fontsize=11)
    ax.set_ylabel('Cepstrum Magnitude', fontsize=11)
    ax.set_title(title, fontsize=11)
    # 标注前 N 个峰值及其表示的功率（幅值平方），仅在可见的 quefrency 范围内
    if top_n and top_n > 0:
        try:
            if qmax is not None:
                visible_mask = (quefrency >= 0) & (quefrency <= qmax)
            else:
                visible_mask = np.ones_like(quefrency, dtype=bool)
            if np.any(visible_mask):
                q_vis = quefrency[visible_mask]
                c_vis = ceps[visible_mask]
                peaks = _fd.dominant_frequencies(q_vis, c_vis, n=top_n)
                for i, (q0, c0) in enumerate(peaks):
                    p0 = c0 * c0
                    ax.scatter([q0], [c0], color=_COLORS[(i+1) % len(_COLORS)], zorder=6)
                    ax.text(q0, c0 * 1.05, f"{q0:.6f} s\nP={p0:.3e}",
                            fontsize=8, ha='center', color=_COLORS[(i+1) % len(_COLORS)])
        except Exception:
            pass

    fig.tight_layout()
    return fig


def envelope_spectrum(freq: np.ndarray, ampl: np.ndarray,
                      fault_freqs: Dict[str, float] = None,
                      n_harmonics: int = 3,
                      xmax: float = None,
                      title: str = 'Envelope Spectrum',
                      top_n: int = 0) -> plt.Figure:
    """
    绘制包络谱，并标注轴承故障特征频率及其谐波。

    Parameters
    ----------
    freq        : 频率轴 (Hz)
    ampl        : 包络幅值
    fault_freqs : bearing.fault_frequencies() 的返回值
    n_harmonics : 标注的谐波次数
    xmax        : X 轴上限
    """
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freq, ampl, color='#37474F', lw=0.8, label='包络谱')
    # default to 1000 Hz
    if xmax is None:
        xmax = 300.0
    ax.set_xlim(0, xmax)

    if fault_freqs:
        fault_colors = {
            'BPFO': '#F44336', 'BPFI': '#2196F3',
            'BSF': '#4CAF50',  'FTF': '#FF9800',
        }
        for fault_type, color in fault_colors.items():
            f0 = fault_freqs.get(fault_type)
            if f0 and f0 > 0:
                for k in range(1, n_harmonics + 1):
                    fk = f0 * k
                    if xmax and fk > xmax:
                        break
                    lw = 1.5 if k == 1 else 0.8
                    ax.axvline(fk, color=color, ls='--', lw=lw, alpha=0.7)
                    if k == 1:
                        ax.text(fk, ax.get_ylim()[1] * 0.9,
                                fault_type, color=color,
                                fontsize=8, ha='center', rotation=90)

        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=c, ls='--', lw=1.5, label=ft)
            for ft, c in fault_colors.items()
            if ft in fault_freqs
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 标注前 N 个峰值及其代表的功率（幅值平方）
    if top_n and top_n > 0:
        try:
            visible_mask = (freq >= 0) & (freq <= xmax) if xmax is not None else np.ones_like(freq, dtype=bool)
            if np.any(visible_mask):
                f_vis = freq[visible_mask]
                a_vis = ampl[visible_mask]
                peaks = _fd.dominant_frequencies(f_vis, a_vis, n=top_n)
                for i, (f0, a0) in enumerate(peaks):
                    p0 = a0 * a0
                    ax.scatter([f0], [a0], color=_COLORS[(i+1) % len(_COLORS)], zorder=6)
                    ax.text(f0, a0 * 1.05, f"{f0:.2f} Hz\nP={p0:.3e}",
                            fontsize=8, ha='center', color=_COLORS[(i+1) % len(_COLORS)])
        except Exception:
            pass

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def gear_mesh_spectrum(freq: np.ndarray, ampl: np.ndarray,
                       mesh_freq: float,
                       rotation_freq: float,
                       n_mesh_harmonics: int = 4,
                       n_sidebands: int = 3,
                       xmax: float = None,
                       title: str = 'Gear Mesh Spectrum') -> plt.Figure:
    """
    绘制齿轮幅值谱并标注啮合频率谐波及边带。
    """
    _ensure_chinese_font()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(freq, ampl, color='#37474F', lw=0.7)
    # default to 1000 Hz
    if xmax is None:
        xmax = 300.0
    ax.set_xlim(0, xmax)

    fm_color = '#D32F2F'
    sb_color = '#1976D2'
    for k in range(1, n_mesh_harmonics + 1):
        fm_k = mesh_freq * k
        if xmax and fm_k > xmax:
            break
        ax.axvline(fm_k, color=fm_color, ls='-', lw=1.2, alpha=0.8)
        ax.text(fm_k, ax.get_ylim()[1] * 0.92, f'{k}×fm',
                color=fm_color, fontsize=7, ha='center')
        for n in range(1, n_sidebands + 1):
            for sign in (+1, -1):
                fsb = fm_k + sign * n * rotation_freq
                if 0 < fsb and (xmax is None or fsb <= xmax):
                    ax.axvline(fsb, color=sb_color, ls=':', lw=0.8, alpha=0.7)

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(title, fontsize=11)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=fm_color, ls='-', lw=1.5, label='Mesh Harmonics'),
        Line2D([0], [0], color=sb_color, ls=':', lw=1.0, label='Rotation Sidebands'),
    ]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    return fig


def compare_signals(signals: List[np.ndarray], fs: float,
                    labels: List[str] = None,
                    unit: str = 'm/s²',
                    title: str = '多信号对比',
                    share_y: bool = False) -> plt.Figure:
    """
    多路信号时域对比图（垂直堆叠）。

    Parameters
    ----------
    signals : 信号列表
    fs      : 采样频率
    labels  : 各信号标签
    unit    : Y 轴单位
    title   : 总标题
    share_y : 是否共享 Y 轴

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(signals)
    if labels is None:
        labels = [f'信号 {i+1}' for i in range(n)]
    _ensure_chinese_font()
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n),
                             sharex=True, sharey=share_y)
    if n == 1:
        axes = [axes]
    for i, (sig, ax, lbl) in enumerate(zip(signals, axes, labels)):
        t = np.arange(len(sig)) / fs
        ax.plot(t, sig, color=_COLORS[i % len(_COLORS)], lw=0.7, label=lbl)
        ax.set_ylabel(f'{lbl}\n({unit})', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    axes[-1].set_xlabel('时间 (s)', fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def dashboard(data: np.ndarray, fs: float,
              title: str = '振动信号分析综合仪表盘',
              fmax: float = None,
              stft_window_s: float = 0.02) -> plt.Figure:
    """
    多子图综合仪表盘（时域波形 + 幅值谱 + STFT 时频图 + 时域统计）。

    Parameters
    ----------
    data           : 振动信号
    fs             : 采样频率 (Hz)
    title          : 仪表盘标题
    fmax           : 频谱显示频率上限
    stft_window_s  : STFT 窗口时长 (s)

    Returns
    -------
    matplotlib.figure.Figure
    """
    from . import freq_domain as _fd_local
    from . import time_freq as _tf_local
    from . import time_domain as td

    _ensure_chinese_font()
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    # rely on _ensure_chinese_font() to pick an available Chinese-capable font
    # ── (1) 时域波形 ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(data)) / fs
    ax1.plot(t, data, color=_COLORS[0], lw=0.7)
    ax1.set_xlabel('时间 (s)', fontsize=10)
    ax1.set_ylabel('幅值', fontsize=10)
    ax1.set_title('时域波形', fontsize=11)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    # ── (2) 幅值谱 ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    freq, ampl = _fd_local.amplitude_spectrum(data, fs)
    ax2.plot(freq, ampl, color=_COLORS[1], lw=0.7)
    if fmax:
        ax2.set_xlim(0, fmax)
    ax2.set_xlabel('频率 (Hz)', fontsize=10)
    ax2.set_ylabel('幅值', fontsize=10)
    ax2.set_title('幅值谱 (FFT)', fontsize=11)

    # ── (3) 功率谱 ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    freq_w, psd = _fd_local.power_spectrum(data, fs)
    ax3.semilogy(freq_w, psd, color=_COLORS[2], lw=0.7)
    if fmax:
        ax3.set_xlim(0, fmax)
    ax3.set_xlabel('频率 (Hz)', fontsize=10)
    ax3.set_ylabel('PSD', fontsize=10)
    ax3.set_title('功率谱密度 (Welch)', fontsize=11)

    # ── (4) STFT 时频图 ───────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    try:
        t_s, f_s, Z = _tf_local.stft(data, fs, window_size=stft_window_s)
        Z_db = 20 * np.log10(Z + 1e-12)
        f_mask = (f_s <= fmax) if fmax else np.ones(len(f_s), bool)
        im = ax4.pcolormesh(t_s, f_s[f_mask], Z_db[f_mask, :],
                            shading='gouraud', cmap='jet')
        fig.colorbar(im, ax=ax4, label='dB', shrink=0.9, pad=0.01)
    except Exception:
        ax4.text(0.5, 0.5, 'STFT 计算失败', ha='center', va='center',
                 transform=ax4.transAxes)
    ax4.set_xlabel('时间 (s)', fontsize=10)
    ax4.set_ylabel('频率 (Hz)', fontsize=10)
    ax4.set_title('STFT 时频谱', fontsize=11)

    # ── 角落统计文本 ──────────────────────────────
    s = td.stats(data)
    stat_txt = (
        f"RMS = {s['rms']:.4g}\n"
        f"峰值 = {s['peak']:.4g}\n"
        f"峰峰值 = {s['peak_to_peak']:.4g}\n"
        f"峭度 = {s['kurtosis']:.4g}\n"
        f"波峰因数 = {s['crest_factor']:.4g}\n"
        f"脉冲因数 = {s['impulse_factor']:.4g}"
    )
    # use the configured sans-serif font (if any) so Chinese glyphs render
    sans = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])[0]
    fig.text(0.97, 0.68, stat_txt, va='top', ha='right',
             fontsize=9, family=sans,
             bbox=dict(boxstyle='round', facecolor='#ECEFF1', alpha=0.8))

    return fig
