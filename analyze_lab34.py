#!/usr/bin/env python3
"""分析脚本：遍历 DATA 中 lab3 和 lab4 的 txt 原始文件，基于 vibsignal 生成图像输出。

用法:
    python analyze_lab34.py

输出:
    在 DATA 目录旁生成 lab3_output 和 lab4_output 文件夹，包含 PNG 图片。
"""
import os
import sys
import re
from pathlib import Path
import numpy as np

# 确保能导入同目录下的 vibsignal 包
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import vibsignal as vs
except Exception:
    # 如果直接从 workspace 根运行，program 目录可能不是 sys.path 的一部分
    sys.path.insert(0, str(HERE))
    import vibsignal as vs


def parse_txt_ts(path: Path):
    """从 txt 文件中解析采样频率和时间序列值。

    返回 (fs, time_array, value_array)
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()

    # 尝试提取采样频率，如 '采样频率：20000Hz' 或 'Sampling rate: 20000Hz'
    fs = None
    for L in lines[:20]:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Hh][Zz]", L)
        if m:
            try:
                fs = float(m.group(1))
                break
            except ValueError:
                pass

    # 找到第一行既能解析为两个浮点数的位置，作为数据起点
    data_start = None
    for i, L in enumerate(lines):
        toks = L.strip().split()
        if len(toks) < 2:
            continue
        try:
            t0 = float(toks[0])
            v0 = float(toks[1])
            data_start = i
            break
        except Exception:
            continue

    if data_start is None:
        raise ValueError(f"未能在 {path} 中找到时间-值数据行")

    times = []
    vals = []
    for L in lines[data_start:]:
        toks = L.strip().split()
        if len(toks) < 2:
            continue
        try:
            tt = float(toks[0])
            vv = float(toks[1])
            times.append(tt)
            vals.append(vv)
        except Exception:
            continue

    times = np.array(times, dtype=float)
    vals = np.array(vals, dtype=float)

    if fs is None:
        # 通过时间间隔估计采样频率
        if len(times) >= 2:
            dt = np.median(np.diff(times))
            fs = 1.0 / dt if dt > 0 else 1.0
        else:
            fs = 1.0

    return float(fs), times, vals


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def analyze_file(path: Path, outdir: Path):
    print(f"Processing: {path}")
    fs, times, vals = parse_txt_ts(path)
    stem = path.stem
    ensure_dir(outdir)

    # # 时域图
    # fig_t = vs.viz.time_waveform(vals, fs, title=f"{stem} (fs={fs:.0f}Hz)")
    # fig_t.savefig(outdir / f"{stem}_time.png")
    # try:
    #     fig_t.clf()
    # except Exception:
    #     pass

    # 幅值谱
    freq, ampl = vs.freq_domain.amplitude_spectrum(vals, fs)
    fig_a = vs.viz.amplitude_spectrum(freq, ampl, title=f"{stem} 幅值谱", top_n=50)
    fig_a.savefig(outdir / f"{stem}_amplitude.png")
    try:
        fig_a.clf()
    except Exception:
        pass

    # # 功率谱
    # fp, psd = vs.freq_domain.power_spectrum(vals, fs)
    # fig_p = vs.viz.power_spectrum(fp, psd, title=f"{stem} 功率谱")
    # fig_p.savefig(outdir / f"{stem}_psd.png")
    # try:
    #     fig_p.clf()
    # except Exception:
    #     pass

    # # 倒频谱
    # try:
    #     q, ceps = vs.freq_domain.cepstrum(vals, fs)
    #     fig_c = vs.viz.cepstrum(q, ceps, title=f"{stem} 倒频谱", top_n=5)
    #     fig_c.savefig(outdir / f"{stem}_cepstrum.png")
    #     try:
    #         fig_c.clf()
    #     except Exception:
    #         pass
    # except Exception as e:
    #     print(f"cepstrum failed: {e}")

    # # 包络谱（若采样率足够高则计算）
    # try:
    #     if fs > 1000:
    #         bp_low = 1000.0
    #         bp_high = min(6000.0, fs / 2.0 - 10.0)
    #         if bp_high <= bp_low:
    #             bp_low = max(10.0, fs * 0.05)
    #             bp_high = max(bp_low + 1.0, fs * 0.45)
    #         fq, fa = vs.bearing.envelope_spectrum(vals, fs, bp_low, bp_high)
    #         fig_env = vs.viz.envelope_spectrum(fq, fa, title=f"{stem} 包络谱", top_n=50)
    #         fig_env.savefig(outdir / f"{stem}_envelope.png")
    #         try:
    #             fig_env.clf()
    #         except Exception:
    #             pass
    # except Exception as e:
    #     print(f"envelope failed: {e}")
    
    # # Dashboard overview (combined plots)
    # try:
    #     fig_d = vs.viz.dashboard(vals, fs, title=f"{stem} Dashboard", fmax=1000.0)
    #     fig_d.savefig(outdir / f"{stem}_dashboard.png")
    #     try:
    #         fig_d.clf()
    #     except Exception:
    #         pass
    # except Exception as e:
    #     print(f"dashboard failed: {e}")


def main():
    # DATA 根目录（相对 program 上一级）
    root = HERE.parent
    data_dir = root / 'DATA'
    if not data_dir.exists():
        print(f"未找到 DATA 目录: {data_dir}")
        return

    tasks = [
        ('lab3轴承', 'lab3_output'),
        ('lab4齿轮', 'lab4_output'),
    ]

    for src_name, out_name in tasks:
        src = data_dir / src_name
        out = data_dir / out_name
        if not src.exists():
            print(f"跳过：未找到 {src}")
            continue
        ensure_dir(out)
        # 处理目录下所有 .txt 文件
        for p in sorted(src.glob('*.txt')):
            try:
                analyze_file(p, out)
            except Exception as e:
                print(f"处理文件失败 {p}: {e}")


if __name__ == '__main__':
    main()
