import numpy as np
import matplotlib.pyplot as plt
from stockwell import st
import os
import struct
import pywt

cmap = plt.get_cmap('rainbow') 

interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval
start = 250000
end = 350000

pwd = os.getcwd()

dd=100
count = 2
filepath = pwd + "//" + "input_data" + "//" + "train1" + "//" + str(dd) + "//" + str(count)

fb = open(filepath, "rb")
x = 0
data1 = []
data2 = []
t = []
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
        t.append(x * 0.0000001)
    x = x + 1

fb.close()

data1 = data1[int(start / interval):int(end / interval)]
data2 = data2[int(start / interval):int(end / interval)]
t = t[int(start / interval):int(end / interval)]

wavelet = 'morl'
c = pywt.central_frequency(wavelet)
fa = np.arange(400000, 20000 - 1, -100)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
power1 = (abs(cfs1))
power2 = (abs(cfs2))

plt.subplot(2,1,1)
plt.imshow(power1,cmap=cmap)
plt.axis('off')

plt.subplot(2,1,2)
plt.imshow(power2,cmap=cmap)
plt.axis('off')

plt.show()

