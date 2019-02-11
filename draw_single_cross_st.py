import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos
import pywt
from stockwell import st

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_'] 

interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval

pwd = os.getcwd()
fb = open(pwd + "//" + "input_data" + "//" + "train1" + "//" + "80" + "//" + "1", "rb")
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

datay1 = datay1[int(250000 / interval):int(350000 / interval)]
datay2 = datay2[int(250000 / interval):int(350000 / interval)]
datax = np.array(range(len(datay1))) * dt

plt.subplot(3,1,1)
plt.ylabel('original signal/V')
plt.plot(datax,datay2,color='b',label = 'ch2')
plt.plot(datax,datay1,color='g',label = 'ch1')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)

fa = 30000 #np.arange(300000, 400000 + 1, 10000)
fmaxmin = int(fa*(len(datay1)*dt))
cfs1 = st.st(datay1, fmaxmin, fmaxmin)[0]
cfs2 = st.st(datay2, fmaxmin, fmaxmin)[0]

power1 = (abs(np.array(cfs1)))
power2 = (abs(np.array(cfs2)))

mean1 = power1.mean()
magnitude1 = power1 / mean1
mean2 =  power2.mean()
magnitude2 = power2 / mean2

plt.subplot(3,1,2)
plt.ylabel('amplitude/V')
plt.plot(datax,magnitude2,color='b',label = 'ch2')
plt.plot(datax,magnitude1,color='g',label = 'ch1')

plt.subplot(3,1,3)
corr = signal.correlate(magnitude1,magnitude2, mode='same',method='fft')
corr_x = (np.array(range(len(corr))) - len(corr) / 2) * dt * interval
plt.plot(corr_x,corr,color='r',label = 'cross correlate')
plt.xlabel('time/s')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()

