import numpy as np
import matplotlib.pyplot as plt

def signal_samples(t):
    return np.sin(1 * np.pi * t) + np.sin(2 * np.pi * t) + np.sin(4 * np.pi * t)

# 设置信号频率、采样频率、频谱精度、采样点数
B = 5.0
f_s = 2 * B
delta_f = 0.01
N = int(f_s / delta_f)
print('信号频率：',B,'Hz')
print('采样频率：',f_s,'Hz\t（必须>2*信号频率）')
print('频谱精度：',delta_f,'\t即每隔多少频率计算一次对应的频率分量')
print('采样点数：',N)
# 采样时间根据：采样点数和采样频率计算得出
T = N / f_s
print('采样时间：',T,'s\t（采样点数/采样频率）')

# 采样
t = np.linspace(0, T, N)
f_t = signal_samples(t)

# 画出信号波形
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].plot(t, f_t)
axes[0].set_xlabel("time (s)")
axes[0].set_ylabel("signal")
axes[1].plot(t, f_t)
axes[1].set_xlim(0, 5)
axes[1].set_xlabel("time (s)")
plt.show()

# 快速傅里叶变换
from scipy import fftpack
F = fftpack.fft(f_t)
# print(abs(F),F.shape)
f = fftpack.fftfreq(N, 1.0/f_s)
mask = np.where(f >= 0)
# print(mask)
fig, axes = plt.subplots(3, 1, figsize=(8, 6))

axes[0].plot(f[mask], np.log(abs(F[mask])), label="real")
axes[0].plot(B, 0, 'r*', markersize=10)
axes[0].set_ylabel("$\log(|F|)$", fontsize=14)

axes[1].plot(f[mask], abs(F[mask])/N, label="real")
axes[1].set_xlim(0, 2.5)
axes[1].set_ylabel("$|F|$", fontsize=14)

axes[2].plot(f[mask], abs(F[mask])/N, label="real")
axes[2].set_xlabel("frequency (Hz)", fontsize=14)
axes[2].set_ylabel("$|F|$", fontsize=14)
plt.show()