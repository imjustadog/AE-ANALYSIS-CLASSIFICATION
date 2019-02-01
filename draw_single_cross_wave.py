import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_'] 

dd = 102
sgn = 1.0
dt = 0.0000001
fs = 10000000.0

pwd = os.getcwd()
fb = open(pwd + "//" + "5" + "//" + str(dd) + "//" + "30", "rb")
x = 0
datax = []
datay1 = []
datay2 = []

while True:
    data = fb.read(4)
    if not data:
        break
    ch1, ch2 = struct.unpack('<HH', data)
    ch1 = (float(ch1) - 8192) / 8192 * 2.5
    ch2 = (float(ch2) - 8192) / 8192 * 2.5
    ch1 = float(ch1)
    ch2 = float(ch2)
    x = x + 1
    datax.append(x * dt)
    datay1.append(ch1)
    datay2.append(ch2)


datay1 = np.array(datay1[250000:350000])
datay2 = np.array(datay2[250000:350000])
datax = np.array(datax[250000:350000])

plt.subplot(2,1,1)
plt.ylabel('original signal/V')
plt.plot(datax,datay2,color='b',label = 'ch2')
plt.plot(datax,datay1,color='g',label = 'ch1')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)

plt.subplot(2,1,2)
corr = signal.correlate(datay1,datay2, mode='same',method='fft')
corr_x = (np.array(range(len(corr))) - len(corr) / 2) * dt
plt.plot(corr_x,corr,color='r',label = 'cross correlate')
plt.xlabel('time/s')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()

