import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos
import pywt

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_',','] 

dd = 102
sgn = 1.0

interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval

pwd = os.getcwd()
fb = open(pwd + "//" + "5" + "//" + str(dd) + "//" + "35", "rb")
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

wavelet = 'morl'
c = pywt.central_frequency(wavelet)
fa = np.arange(400000, 120000-1, -20000)
scales = np.array(float(c)) * fs / np.array(fa)

[cfs1,frequencies1] = pywt.cwt(datay1,scales,wavelet,dt)
[cfs2,frequencies2] = pywt.cwt(datay2,scales,wavelet,dt)
magnitude1 = (abs(cfs1)) ** 2
magnitude2 = (abs(cfs2)) ** 2

corr_show = []
for i in range(len(magnitude1)):
    mean1 = magnitude1[i].mean()
    magnitude1[i] = magnitude1[i] / mean1
    mean2 = magnitude2[i].mean()
    magnitude2[i] = magnitude2[i] / mean2
    temp = signal.correlate(magnitude1[i],magnitude2[i], mode='same',method='fft')
    meanc = temp.mean()
    corr_show.append(temp / meanc) 

##corr = np.array(corr)
##
##E=207 * pow(10,9) #203#207
##p=7.86 * 1000 #7.93#7.86
##o=0.27
##h=0.002
##
##param = E * h * h * pi * pi / 3.0 / p / (1.0 - o * o)
##c = pow(param * pow(freq,2),0.25)
##time = sgn * (100.0 - dd) * 2.0 / 100.0 / c * 1000.0 / 1.5
##
##fig = plt.figure(figsize=(5,3))
##
##plt.plot(freq/1000.0,corr / 1000.0 * c,label = 'dd')
##
##print sum(corr / 1000.0 * c) / len(freq)
##

for index,item in enumerate(corr_show):
    print frequencies1[index] / 1000,max(item),(np.where(item == max(item))[0][0]-len(item) / 2 ) * dt * 1000
    plt.plot(item,marker = markers_freq[index], label = str(frequencies1[index] / 1000) + 'kHz')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)
plt.show()

#tempsum = np.sum(corr_show,axis = 0)
#print ((np.where(tempsum == max(tempsum))[0][0] - 10000 / interval ) * interval * dt * 1000)
