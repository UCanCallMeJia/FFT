# 傅立叶变换— Fast Fourier Transform 

傅里叶变换（**DFT**）将难以处理的时域信号转换成易于分析的频域信号（信号的频谱），同时也可以利用傅里叶反变换将频域信号转化为时域信号。  
快速傅里叶变换（**FFT**）是**DFT**的快速算法，**FFT**的计算结果和**DFT**是完全等价的，相比之下只是运算量下降。
## 采样定理
>   **采样定理**: 如果以超过函数最高频率的两倍的取样率来获得样本，连续的带限函数可以完全地从它的样本集来恢复。采样频率为Fs，信号频率为F，采样点数为N，频域抽样间隔为F0，即频谱精度。模拟信号经过A/D转换变为数字信号的过程称为采样。为保证采样后信号的频谱形状不失真，采样频率必须大于信号中最高频率成分的2倍，这称之为采样定理。 假设采样频率为fs，采样点数为N，那么FFT结果就是一个N点的复数，每一个点就对应着一个频率点，某一点n(n从1开始)表示的频率为：fn=(n-1)*fs/N。 
## 实验
被采样的信号为：    sin(pi*t)+sin(2*pi*t)+sin(4*pi*t)  
![Original Signal](https://github.com/UCanCallMeJia/FFT/blob/master/FFT.png)  
所以该信号包含了三个频率分别为2Hz, 1Hz, 0.5Hz的正弦信号。     
信号频率为 F=2Hz  
其中采样频率我们取为 Fs=10Hz,  
频谱精度我们取为 f0=0,01 ,  
则可以计算采样点数为: 1000 ,  
采样时间可以计算：10s 。  

