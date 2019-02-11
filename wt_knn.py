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

fig_size = 40

pwd = os.getcwd()

dd=60
count = 0
filepath = pwd + "//" + "input_data" + "//" + "train1" + "//" + str(dd) + "//" + str(count)
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

data1 = data1[int(start / interval):int(end / interval)]
data2 = data2[int(start / interval):int(end / interval)]

wavelet = 'morl'
c = pywt.central_frequency(wavelet)
fa = np.arange(400000, 20000 - 1, -10000)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
power1 = (abs(cfs1)) ** 2
power2 = (abs(cfs2)) ** 2

#print(len(power1),len(power1[0]))

length_now = len(power1[0])
power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
power1 = np.log10(np.mean(power1,axis=2))
power2 = np.log10(np.mean(power2,axis=2))

mx = power1.max()
mn = power1.min()
power1 = (power1-mn) / (mx-mn) * 255.0
power1 = np.floor(power1)

mx = power2.max()
mn = power2.min()
power2 = (power2-mn) / (mx-mn) * 255.0
power2 = np.floor(power2)

k1=power1




dd=140
count = 8
filepath = pwd + "//" + "input_data" + "//" + "other" + "//" + "drop" + "//" + str(count)
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

data1 = data1[int(start / interval):int(end / interval)]
data2 = data2[int(start / interval):int(end / interval)]

wavelet = 'morl'
c = pywt.central_frequency(wavelet)
fa = np.arange(400000, 20000 - 1, -10000)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
power1 = (abs(cfs1)) ** 2
power2 = (abs(cfs2)) ** 2

#print(len(power1),len(power1[0]))

length_now = len(power1[0])
power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
power1 = np.log10(np.mean(power1,axis=2))
power2 = np.log10(np.mean(power2,axis=2))

mx = power1.max()
mn = power1.min()
power1 = (power1-mn) / (mx-mn) * 255.0
power1 = np.floor(power1)

mx = power2.max()
mn = power2.min()
power2 = (power2-mn) / (mx-mn) * 255.0
power2 = np.floor(power2)

k2=power1

dis=np.array(abs(k1-k2))
print(sum(dis.flatten()))


plt.subplot(2,1,1)
plt.imshow(k1,cmap=cmap)
plt.axis('off')

plt.subplot(2,1,2)
plt.imshow(k2,cmap=cmap)
plt.axis('off')

plt.show()
