import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_'] 

dd = 42
sgn = 1.0
dt = 0.0000001
fs = 10000000.0

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

plt.subplot(3,1,1)
plt.ylabel('original signal/V')
plt.plot(datax,datay2,color='b',label = 'ch2')
plt.plot(datax,datay1,color='g',label = 'ch1')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)

count1 = len(datay1)
count2 = len(datay2)

fftnum = 500#64#512#1024#8192
std = 125#16#128#256#1500

axis_xf = [11]
freq = np.array([fs * i / fftnum for i in axis_xf])

win = signal.gaussian(fftnum, std)

basis = []
for k in [11]:
    basis.append([complex(cos(2*pi/fftnum*k*n),sin(2*pi/fftnum*k*n)) for n in range(fftnum)])

basis = np.transpose(basis)

index = 0
interval = 5
magnitude1 = []
magnitude2 = []
while True:
    data1 = np.array(datay1[index:index + fftnum])
    data1 = np.multiply(data1,win)
    data1 = np.dot(data1,basis)
    data1 = np.abs(data1) / fftnum * 2
    magnitude1.append(data1)
    if index + fftnum <= count2:
        data2 = np.array(datay2[index:index + fftnum])
        data2 = np.multiply(data2,win)
        data2 = np.dot(data2,basis)
        data2 = np.abs(data2) / fftnum * 2
        magnitude2.append(data2) 
    index += interval
    if index + fftnum > count1:
        break

magnitude1 = np.array(magnitude1).T
magnitude2 = np.array(magnitude2).T

#mean1 = magnitude1[0].mean()
#magnitude1[0] = magnitude1[0]/ mean1
#mean2 = magnitude2[0].mean()
#magnitude2[0] = magnitude2[0]/mean2

datax = datax[0:len(magnitude1[0])*5:5]

plt.subplot(3,1,2)
plt.ylabel('amplitude/V')
plt.plot(datax,magnitude2[0],color='b',label = 'ch2')
plt.plot(datax,magnitude1[0],color='g',label = 'ch1')

plt.subplot(3,1,3)
corr = signal.correlate(magnitude1[0],magnitude2[0], mode='same',method='fft')
corr_x = (np.array(range(len(corr))) - len(corr) / 2) * dt * 5
plt.plot(corr_x,corr,color='r',label = 'cross correlate')
plt.xlabel('time/s')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()

