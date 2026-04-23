import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from .solution import hilbert_transform
import os

data_slice=slice(15, 1000)
N=1485
T=1.0/1500.0
savefig_path = "figs/"

def test():
    print("This is a test function in visible.py")

def draw_three_values(temp1, temp2, temp3,name):
    """绘制三个信号的时域图"""
    print("only show",data_slice,"size of data")
    if name is None:
        name = ''
    temp1=temp1[data_slice]
    temp2=temp2[data_slice]
    temp3=temp3[data_slice]
    time_axis = np.arange(len(temp1)) * T
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    plt.plot(time_axis, temp1)
    plt.title('Signal AI-03 Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(time_axis, temp2)
    plt.title('Signal AI-04 Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(time_axis, temp3)
    plt.title('Signal AI-05 Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(savefig_path, name+'time_domain.png'))
    print("Time domain plot saved to:", os.path.join(savefig_path, name+'time_domain.png'))
    # plt.show()

def draw_three_values_sub(temp1, temp2,name):
    """绘制三个信号的时域图"""
    print("only show",data_slice,"size of data")
    if name is None:
        name = ''
    temp1=temp1[data_slice]
    temp2=temp2[data_slice]
    time_axis = np.arange(len(temp1)) * T
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 3, 1)
    plt.plot(time_axis, temp1)
    plt.title('Signal AI8-06 Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(time_axis, temp2)
    plt.title('Signal AI8-07 Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(savefig_path, name+'time_domain.png'))
    print("Time domain plot saved to:", os.path.join(savefig_path, name+'time_domain.png'))
    # plt.show()

def draw_plot_fft(temp1,name):
    """绘制三个信号的FFT变换//以及时域图"""
    print("only show",data_slice,"size of data")
    if name is None:
        name = ''
    temp1=temp1[data_slice]
    yf1=fft(temp1)
    xf=fftfreq(N,T)[:N//2]
    dic_signals = ['AI-03']
    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # for i in range(3):
    #     for j in range(2):
    #         if j==0:
    #             axs[j, i].plot(xf, np.abs([yf1, yf2, yf3][i][0:N//2]))
    #         else:
    #             axs[j, i].plot([temp1, temp2, temp3][i])
    #         axs[j, i].grid()
    #         axs[j, i].set_xlabel('Frequency (Hz)' if j==0 else 'Time (s)')
    #         axs[j, i].set_ylabel('Magnitude' if j==0 else 'Amplitude')
    #         axs[j, i].set_title(f'Signal AI-0{i+3} {"FFT Spectrum" if j==0 else "Time Domain"}')
    fig, axs = plt.subplots(figsize=(24, 6))
    for i in range(1):
        magnitudes = np.abs([yf1][i][:N//2])
        if i < 2:
            mask = xf < 75
            axs.plot(xf[mask], magnitudes[mask])
        else:
            axs.plot(xf, magnitudes)
        axs.grid()
        axs.set_xlabel('Frequency (Hz)')
        axs.set_ylabel('Magnitude')
        axs.set_title(f'Signal {dic_signals[i]} FFT Spectrum')
        if i < 2:  # 只在前两个图中标注峰值
            # 找到局部峰值：比左右邻居都大的点
            peaks = []
            for j in range(1, len(magnitudes)-1):
                if magnitudes[j] > magnitudes[j-1] and magnitudes[j] > magnitudes[j+1]:
                    peaks.append((magnitudes[j], j))
            peaks.sort(reverse=True)  # 按幅度降序排序
            top_indices = [idx for mag, idx in peaks[:3]]  # 取前三个最大的局部峰值
            for j, idx in enumerate(top_indices):
                freq = xf[idx]
                if freq < 200:  # 只标注频率小于200的峰值
                    mag = magnitudes[idx]
                    # 添加垂直和水平定位线
                    axs.axvline(x=freq, color='red', linestyle='--', alpha=0.7)
                    axs.axhline(y=mag, color='red', linestyle='--', alpha=0.7)
                    # 在下面标注峰值
                    # axs[i].text(freq, mag - 0.05 * np.max(magnitudes), f'{freq:.0f} Hz\n{mag:.0f}', 
                    #             ha='center', va='top', fontsize=9, color='red')
                    axs.text(freq, mag+0.05* np.max(magnitudes), f'{freq:.1f} Hz', 
                                ha='center', va='top', fontsize=5, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(savefig_path, name+'fft_analysis.png'))
    print("FFT analysis plot saved to:", os.path.join(savefig_path, name+'fft_analysis.png'))
    # plt.show()

def draw_three_plot_fft(temp1, temp2, temp3,name):
    """绘制三个信号的FFT变换//以及时域图"""
    """第一个实验：不平衡"""
    print("only show",data_slice,"size of data")
    if name is None:
        name = ''
    temp1=temp1[data_slice]
    temp2=temp2[data_slice]
    temp3=temp3[data_slice]
    yf1=fft(temp1)
    yf2=fft(temp2)
    yf3=fft(temp3)
    xf=fftfreq(N,T)[:N//2]

    """第一个实验：不平衡"""
    dic_signals = ['AI-03','AI-04','AI-05']

    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # for i in range(3):
    #     for j in range(2):
    #         if j==0:
    #             axs[j, i].plot(xf, np.abs([yf1, yf2, yf3][i][0:N//2]))
    #         else:
    #             axs[j, i].plot([temp1, temp2, temp3][i])
    #         axs[j, i].grid()
    #         axs[j, i].set_xlabel('Frequency (Hz)' if j==0 else 'Time (s)')
    #         axs[j, i].set_ylabel('Magnitude' if j==0 else 'Amplitude')
    #         axs[j, i].set_title(f'Signal AI-0{i+3} {"FFT Spectrum" if j==0 else "Time Domain"}')
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    for i in range(3):
        magnitudes = np.abs([yf1, yf2, yf3][i][:N//2])
        if i < 2:
            mask = xf < 75
            axs[i].plot(xf[mask], magnitudes[mask])
        else:
            axs[i].plot(xf, magnitudes)
        axs[i].grid()
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Magnitude')
        axs[i].set_title(f'Signal {dic_signals[i]} FFT Spectrum')
        if i < 2:  # 只在前两个图中标注峰值
            # 找到局部峰值：比左右邻居都大的点
            peaks = []
            for j in range(1, len(magnitudes)-1):
                if magnitudes[j] > magnitudes[j-1] and magnitudes[j] > magnitudes[j+1]:
                    peaks.append((magnitudes[j], j))
            peaks.sort(reverse=True)  # 按幅度降序排序
            top_indices = [idx for mag, idx in peaks[:3]]  # 取前三个最大的局部峰值
            for j, idx in enumerate(top_indices):
                freq = xf[idx]
                if freq < 200:  # 只标注频率小于200的峰值
                    mag = magnitudes[idx]
                    # 添加垂直和水平定位线
                    axs[i].axvline(x=freq, color='red', linestyle='--', alpha=0.7)
                    axs[i].axhline(y=mag, color='red', linestyle='--', alpha=0.7)
                    # 在下面标注峰值
                    # axs[i].text(freq, mag - 0.05 * np.max(magnitudes), f'{freq:.0f} Hz\n{mag:.0f}', 
                    #             ha='center', va='top', fontsize=9, color='red')
                    axs[i].text(freq, mag+0.05* np.max(magnitudes), f'{freq:.1f} Hz', 
                                ha='center', va='top', fontsize=5, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(savefig_path, name+'fft_analysis.png'))
    print("FFT analysis plot saved to:", os.path.join(savefig_path, name+'fft_analysis.png'))
    # plt.show()

def sub_draw_three_plot_fft(temp1, temp2,name):
    """绘制三个信号的FFT变换//以及时域图"""
    """第二个实验：不对中"""
    print("only show",data_slice,"size of data")
    if name is None:
        name = ''
    temp1=temp1[data_slice]
    temp2=temp2[data_slice]

    yf1=fft(temp1)
    yf2=fft(temp2)

    xf=fftfreq(N,T)[:N//2]
    
    dic_signals = ['AI8-06','AI8-07']
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))
    for i in range(2):
        magnitudes = np.abs([yf1, yf2][i][:N//2])
        if i < 2:
            mask = xf < 200
            axs[i].plot(xf[mask], magnitudes[mask])
        else:
            axs[i].plot(xf, magnitudes)
        axs[i].grid()
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Magnitude')
        axs[i].set_title(f'Signal {dic_signals[i]} FFT Spectrum')
        if i < 2:  # 只在前两个图中标注峰值
            # 找到局部峰值：比左右邻居都大的点
            peaks = []
            for j in range(1, len(magnitudes)-1):
                if magnitudes[j] > magnitudes[j-1] and magnitudes[j] > magnitudes[j+1]:
                    peaks.append((magnitudes[j], j))
            peaks.sort(reverse=True)  # 按幅度降序排序
            top_indices = [idx for mag, idx in peaks[:3]]  # 取前三个最大的局部峰值
            for j, idx in enumerate(top_indices):
                freq = xf[idx]
                if freq < 500:  # 只标注频率小于500的峰值
                    mag = magnitudes[idx]
                    # 添加垂直和水平定位线
                    axs[i].axvline(x=freq, color='red', linestyle='--', alpha=0.7)
                    axs[i].axhline(y=mag, color='red', linestyle='--', alpha=0.7)
                    # 在下面标注峰值
                    # axs[i].text(freq, mag - 0.05 * np.max(magnitudes), f'{freq:.0f} Hz\n{mag:.0f}', 
                    #             ha='center', va='top', fontsize=9, color='red')
                    axs[i].text(freq, mag+0.05* np.max(magnitudes), f'{freq:.1f} Hz', 
                                ha='center', va='top', fontsize=5, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(savefig_path, name+'fft_analysis.png'))
    print("FFT analysis plot saved to:", os.path.join(savefig_path, name+'fft_analysis.png'))
    # plt.show()


def draw_trajectory(temp1, temp2, name):
    """绘制XY轨迹图"""
    print("only show",data_slice,"size of data")
    if name is None:
        name = ''
    temp1=temp1[data_slice]
    temp2=temp2[data_slice]
    plt.figure(figsize=(8, 8))
    plt.plot(temp1, temp2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('XY Trajectory')
    plt.grid()
    plt.savefig(os.path.join(savefig_path, name+'xy_trajectory.png'))
    print("XY trajectory plot saved to:", os.path.join(savefig_path, name+'xy_trajectory.png'))
    # plt.show()

def autocorrelation(data):
    """计算信号的自相关"""
    data = np.asarray(data, dtype=float)
    autocorr = np.correlate(data, data, mode='full')
    return autocorr

def crosscorrelation(data1, data2):
    """计算两个信号的互相关"""
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    crosscorr = np.correlate(data1, data2, mode='full')
    return crosscorr

def plot_autocorrelation(data, name):
    """绘制信号的自相关图像"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    autocorr = autocorrelation(data)
    lags = np.arange(-len(data) + 1, len(data))
    n=len(lags)
    plt.figure(figsize=(10, 6))
    plt.plot(lags[n//2:], autocorr[n//2:])  # 只显示正延迟部分
    plt.title('自相关分析')
    plt.xlabel('延迟 (lags)')
    plt.ylabel('相关性')
    plt.grid(True)
    plt.savefig(os.path.join(savefig_path, name + '_autocorrelation.png'))
    print("自相关图像已保存到:", os.path.join(savefig_path, name + '_autocorrelation.png'))
    # plt.show()

def plot_crosscorrelation(data1, data2, name):
    """绘制两个信号的互相关图像"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    crosscorr = crosscorrelation(data1, data2)
    lags = np.arange(-len(data1) + 1, len(data1))
    n=len(lags)
    plt.figure(figsize=(10, 6))
    plt.plot(lags[n//2:], crosscorr[n//2:])  # 只显示正延迟部分
    plt.title('互相关分析')
    plt.xlabel('延迟 (lags)')
    plt.ylabel('相关性')
    plt.grid(True)
    plt.savefig(os.path.join(savefig_path, name + '_crosscorrelation.png'))
    print("互相关图像已保存到:", os.path.join(savefig_path, name + '_crosscorrelation.png'))
    # plt.show()

def plot_draw_hilbert_transform(data, name):
    """绘制信号的希尔伯特变换结果"""
    analytic_signal, amplitude, phase, frequency = hilbert_transform(data)
    envelope = abs(analytic_signal)
    inv_envelope = -envelope
    plt.figure(figsize=(12, 6))
    plt.plot(analytic_signal, label='real')
    plt.plot(envelope, label='envelope', color='red')
    plt.show()