import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos
import pywt

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_'] 

dd = 102
sgn = 1.0

interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval

pwd = os.getcwd()
fb = open(pwd + "//" + "5" + "//" + str(dd) + "//" + "33", "rb")
x = 0
datax = []
datay1 = []
datay2 = []

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
        datay1.append(ch1)
        datay2.append(ch2)
        datax.append(x * dt)
    x = x + 1

fb.close()

datay1 = datay1[250000 / interval:350000 / interval]
datay2 = datay2[250000 / interval:350000 / interval]
datax = np.array(range(len(datay1))) * dt

plt.subplot(3,1,1)
plt.ylabel('original signal/V')
plt.plot(datax,datay2,color='b',label = 'ch2')
plt.plot(datax,datay1,color='g',label = 'ch1')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)

wavelet = 'morl'
c = pywt.central_frequency(wavelet)
fa = [220000] #np.arange(400000, 0, -20000)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(datay1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(datay2,scales,wavelet,dt)
magnitude1 = (abs(cfs1)) ** 2
magnitude2 = (abs(cfs2)) ** 2

mean1 = magnitude1[0].mean()
magnitude1[0] = magnitude1[0]/ mean1
mean2 = magnitude2[0].mean()
magnitude2[0] = magnitude2[0]/mean2

plt.subplot(3,1,2)
plt.ylabel('amplitude/V')
plt.plot(datax,magnitude2[0],color='b',label = 'ch2')
plt.plot(datax,magnitude1[0],color='g',label = 'ch1')

plt.subplot(3,1,3)
corr = signal.correlate(magnitude1[0],magnitude2[0], mode='same',method='fft')
corr_x = (np.array(range(len(corr))) - len(corr) / 2) * dt * interval
plt.plot(corr_x,corr,color='r',label = 'cross correlate')
plt.xlabel('time/s')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()

