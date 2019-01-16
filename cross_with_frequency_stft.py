import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_'] 

dd = 142
sgn = 1.0
dt = 0.0000001
fs = 10000000.0

pwd = os.getcwd()
fb = open(pwd + "//" + "5" + "//" + str(dd) + "//" + "31", "rb")
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


datay1 = datay1[250000:350000]
datay2 = datay2[260000:340000]

count1 = len(datay1)
count2 = len(datay2)

fftnum = 500#64#512#1024#8192
std = 125#16#128#256#1500

axis_xf = range(3,21)
freq = np.array([fs * i / fftnum for i in axis_xf])

win = signal.gaussian(fftnum, std)

basis = []
for k in range(3,21):
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
    data1 = np.abs(data1)
    magnitude1.append(data1)
    if index + fftnum <= count2:
        data2 = np.array(datay2[index:index + fftnum])
        data2 = np.multiply(data2,win)
        data2 = np.dot(data2,basis)
        data2 = np.abs(data2)
        magnitude2.append(data2)
    index += interval
    if index + fftnum > count1:
        break

temp = signal.correlate(magnitude1,magnitude2, mode='valid',method='fft')
corr = ((np.where(temp == max(temp))[0][0] - 10000 / interval ) * interval * dt * 1000)
print corr


magnitude1 = np.array(magnitude1).T
magnitude2 = np.array(magnitude2).T
corr_show = []
for i in range(len(magnitude1)):
    mean1 = magnitude1[i].mean()
    magnitude1[i] = magnitude1[i] / mean1
    mean2 = magnitude2[i].mean()
    magnitude2[i] = magnitude2[i] / mean2
    temp = signal.correlate(magnitude1[i],magnitude2[i], mode='valid',method='fft')
    corr_show.append(temp) 

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
    print axis_xf[index] * 20,max(item),((np.where(item == max(item))[0][0] - 10000 / interval ) * interval * dt * 1000)
    #plt.figure()
    plt.plot(item,marker = markers_freq[index], label = str(axis_xf[index] * 20) + 'kHz')
    #plt.show()
    
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)
plt.show()

#tempsum = np.sum(corr_show,axis = 0)
#print ((np.where(tempsum == max(tempsum))[0][0] - 10000 / interval ) * interval * dt * 1000)
