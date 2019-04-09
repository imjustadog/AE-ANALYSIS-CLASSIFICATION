import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi

import pywt
import struct

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_',','] 
fig = plt.figure(figsize=(5,5))

sgn = 1.0
interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval
start = 250000
end = 350000

result = []
pwd = os.getcwd()

dict_fitting = {}

for dd in np.arange(40,160,10):
    count = 0
    record = []
    while True:
        filepath = pwd + "//" + "input_data" + "//" + "5" + "//" + str(dd) + "//" + str(count)

        if os.path.isfile(filepath) == False:
            break
        
        with open(filepath, "rb") as fb:
            data = fb.read()

        ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
        ch1ch2 = np.array(ch1ch2)
        ch1ch2 = (ch1ch2-8192)*2.5/8192

        datay1 = ch1ch2[::2]
        datay2 = ch1ch2[1::2]
        
        data1 = datay1[start:end:interval]
        data2 = datay2[start:end:interval]

        wavelet = 'morl'
        c = pywt.central_frequency(wavelet)
        fa = np.arange(20000, 400000 + 1, 10000)
        scales = np.array(float(c)) * fs / np.array(fa)

        [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
        [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
        power1 = (abs(cfs1)) ** 2
        power2 = (abs(cfs2)) ** 2

        for i,f in enumerate(fa):
            mean1 = power1[i].mean()
            power1[i] = power1[i] / mean1
            mean2 =  power2[i].mean()
            power2[i] = power2[i] / mean2
            temp = signal.correlate(power1[i],power2[i], mode='same',method='fft')
            corr = (np.where(temp == max(temp))[0][0]-len(temp) / 2 ) * dt * 1000
            dict_fitting[f] = [dd, corr]

        print(filepath)

        count = count + 1
        if count > 20:
            break

plt.subplot(2,1,1)
x = []
y = []
for index,item in enumerate(result):
    for yi in item:
        xi = 120 - index * 20
        x.append(xi)
        y.append(yi)
        plt.plot(xi, yi, '+')

f = np.polyfit(x,y,1)
xf = [120 - i * 20 for i in range(0,13)]
yf = [xi * f[0] + f[1] for xi in xf]
plt.plot(xf,yf,'r-')
plt.ylabel('time difference/ms')
print(f)

plt.subplot(2,1,2)
error = []
for index,item in enumerate(result):
    for yi in item:
        xi = 120 - index * 20
        xf = (yi - f[1])/f[0]
        error.append(abs(xf - xi))
        plt.plot(xi,xf,'+')
plt.ylabel('calculation result/cm')
print(np.mean(error))

plt.xlabel('distance difference/cm')
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()
