import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi

import pywt
import struct

cmap = plt.get_cmap('rainbow') # this may fail on older versions of matplotlib
interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval
start = 250000
end = 350000

fig_size = 20

pwd = os.getcwd()

dd=80
count = 2
filepath = pwd + "//" + "input_data" + "//" + "train1" + "//" + str(dd) + "//" + str(count)

with open(filepath, "rb") as fb:
    data = fb.read()

ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
ch1ch2 = np.array(ch1ch2)
ch1ch2 = (ch1ch2-8192)*2.5/8192

datay1 = ch1ch2[::2]
datay2 = ch1ch2[1::2]
        
data1 = datay1[start:end:interval]
data2 = datay2[start:end:interval]

wavelet = 'db8'
c = pywt.central_frequency(wavelet)
fa = np.arange(400000, 20000, -1000)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
power1 = (abs(cfs1))
power2 = (abs(cfs2))

#print(len(power1),len(power1[0]))

length_now = len(power1[0])
power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
power1 = np.mean(power1,axis=2)
power2 = np.mean(power2,axis=2)

power1 = power1.T
power2 = power2.T

length_now = len(power1[0])
power1 = np.reshape(power1,(fig_size,fig_size,int(length_now/fig_size)))
power2 = np.reshape(power2,(fig_size,fig_size,int(length_now/fig_size)))
power1 = np.mean(power1,axis=2)
power2 = np.mean(power2,axis=2)

power1 = power1.T
power2 = power2.T

power1 = np.log10(power1)
power2 = np.log10(power2)

mx = power1.max()
mn = power1.min()
power1 = (power1-mn) / (mx-mn) * 255.0
power1 = np.floor(power1)

mx = power2.max()
mn = power2.min()
power2 = (power2-mn) / (mx-mn) * 255.0
power2 = np.floor(power2)

plt.subplot(2,1,1)
plt.imshow(power1,cmap=cmap)
plt.axis('off')

plt.subplot(2,1,2)
plt.imshow(power2,cmap=cmap)
plt.axis('off')

plt.show()
