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


result = []
pwd = os.getcwd()

result = []

#savepath = pwd + "//" + "wt_1.txt"
#fr = open(savepath, "w")

curvename = []

dd=60
count = 0
record = []

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
fa = np.arange(400000, 20000 - 1, -20000)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
power1 = (abs(cfs1)) ** 2
power2 = (abs(cfs2)) ** 2

#print(len(power1),len(power1[0]))

time = np.arange(start,end,interval)
time = [i * 0.0000001 for i in time]

plt.subplot(2,1,1)
ax = plt.gca()
cax = ax.contourf(time, frequencies1, np.log10(power1), extend='both',cmap=cmap)

plt.subplot(2,1,2)
ax = plt.gca()
cax = ax.contourf(time, frequencies2, np.log10(power2), extend='both',cmap=cmap)

#cbar = plt.colorbar(cax)
plt.show()
