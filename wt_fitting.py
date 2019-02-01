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

result = []
pwd = os.getcwd()

result = []

#savepath = pwd + "//" + "wt_1.txt"
#fr = open(savepath, "w")

for dd in np.arange(42,163,10):
    count = 0
    record = []
    while True:
        filepath = pwd + "//" + "input_data" + "//" + "5" + "//" + str(dd) + "//" + str(count)

        if os.path.isfile(filepath) == False:
            break
        
        fb = open(filepath, "rb")
        x = 0
        data1 = []
        data2 = []

        while True:
            data = fb.read(4)
            if not data:
                break
            if x % interval == 0:
                ch1, ch2 = struct.unpack('<HH', data)
                ch1 = (float(ch1) - 8192) / 8192 * 2.5
                ch2 = (float(ch2) - 8192) / 8192 * 2.5
                ch1 = float(ch1)
                ch2 = float(ch2)
                data1.append(ch1)
                data2.append(ch2)
            x = x + 1

        fb.close()

        data1 = data1[int(250000 / interval):int(350000 / interval)]
        data2 = data2[int(250000 / interval):int(350000 / interval)]

        wavelet = 'morl'
        c = pywt.central_frequency(wavelet)
        fa = [320000] #np.arange(400000, 0, -20000)
        scales = np.array(float(c)) * fs / np.array(fa)

        [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
        [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
        power1 = (abs(cfs1)) ** 2
        power2 = (abs(cfs2)) ** 2

        corr = []
        for i in range(len(power1)):
            mean1 = power1[i].mean()
            power1[i] = power1[i] / mean1
            mean2 =  power2[i].mean()
            power2[i] = power2[i] / mean2
            temp = signal.correlate(power1[i],power2[i], mode='same',method='fft')
            corr.append((np.where(temp == max(temp))[0][0]-len(temp) / 2 ) * dt * 1000)

        print(corr)
        record.append(corr[0])

        count = count + 1
        if count > 20:
            break
    result.append(record)

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
