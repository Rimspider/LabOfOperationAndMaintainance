"""
vibsignal — 机械振动信号处理库
适用于转轴、滚动轴承、齿轮系统的振动监测与故障诊断

模块结构
--------
vibsignal.filters     — 滤波器（低通、高通、带通、带阻、包络）
vibsignal.time_domain — 时域分析（统计特征、峭度、波形指标、相关分析）
vibsignal.freq_domain — 频域分析（FFT 幅值谱、功率谱、倒频谱）
vibsignal.time_freq   — 时频分析（STFT、小波变换、Hilbert 包络谱）
vibsignal.orbit       — 轴心轨迹分析
vibsignal.bearing     — 滚动轴承故障特征频率计算与诊断
vibsignal.gear        — 齿轮系统故障特征分析
vibsignal.viz         — 可视化（时域图、频谱图、时频图、轴心轨迹等）
vibsignal.utils       — 工具函数（信号生成、加窗、单位换算）
"""

from . import filters
from . import time_domain
from . import freq_domain
from . import time_freq
from . import orbit
from . import bearing
from . import gear
from . import viz
from . import utils

__version__ = "1.0.0"
__all__ = [
    "filters", "time_domain", "freq_domain",
    "time_freq", "orbit", "bearing", "gear", "viz", "utils",
]
