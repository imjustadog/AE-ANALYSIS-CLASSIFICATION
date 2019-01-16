import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi,sin,cos

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_',','] 
fig = plt.figure(figsize=(5,3))

num = np.array([1,0,-1])
den = np.array([1,-1.9039,0.9195])

sgn = 1.0
interval = 5

dt = 0.0000001
fs = 10000000.0

fftnum = 20#64#512#1024#8192
std = 5#16#128#256#1500

axis_xf = range(3,21)
freq = np.array([fs * i / fftnum for i in axis_xf])

win = signal.gaussian(fftnum, std)

result = []

pwd = os.getcwd()

for dd in np.arange(42,163,10):
    count = 30
    record = []
    while True:
        filepath = pwd + "//" + "5" + "//" + str(dd) + "//" + str(count)

        if os.path.isfile(filepath) == False:
            break
        
        fb = open(filepath, "rb")
        x = 0
        data1 = []
        data2 = []

        while True:
            data = fb.read(4)
            if not data:
                break
            ch1, ch2 = struct.unpack('<HH', data)
            ch1 = (float(ch1) - 8192) / 8192 * 2.5
            ch2 = (float(ch2) - 8192) / 8192 * 2.5
            ch1 = float(ch1)
            ch2 = float(ch2)
            data1.append(ch1)
            data2.append(ch2)
            x = x + 1

        fb.close()

        datay1 = data1[250000:350000]
        datay2 = data2[260000:340000]

        #datay1 = signal.lfilter(num,den,data1[250000:350000])
        #datay2 = signal.lfilter(num,den,data2[260000:340000])

        count1 = len(datay1)
        count2 = len(datay2)
        
        index = 0
        magnitude1 = []
        magnitude2 = []
        while True:
            data1 = np.array(datay1[index:index + fftnum])
            data1 = np.multiply(data1,win)
            data1 = sum(np.abs(data1))
            magnitude1.append(data1)
            if index + fftnum <= count2:
                data2 = np.array(datay2[index:index + fftnum])
                data2 = np.multiply(data2,win)
                data2 = sum(np.abs(data2))
                magnitude2.append(data2)
            index += interval
            if index + fftnum > count1:
                break

        temp = signal.correlate(magnitude1,magnitude2, mode='valid',method='fft')
        corr = ((np.where(temp == max(temp))[0][0] - 10000 / interval ) * interval * dt * 1000)
        print corr
        record.append(corr)

##        magnitude1 = np.array(magnitude1).T
##        magnitude2 = np.array(magnitude2).T
##        corr = []
##        for i in range(len(magnitude1)):
##            mean1 = magnitude1[i].mean()
##            magnitude1[i] = magnitude1[i] / mean1
##            mean2 = magnitude2[i].mean()
##            magnitude2[i] = magnitude2[i] / mean2
##            temp = signal.correlate(magnitude1[i],magnitude2[i], mode='same',method='fft')
##            corr.append((np.where(temp == max(temp))[0][0] - len(temp) / 2 ) * interval * dt * 1000)
##        print corr
##        record.append(corr[0])
        
##
##        tdoa = max(corr,key = lambda x:x[0])
##        print tdoa[2],(tdoa[1] - 10000 / interval ) * interval * dt * 1000
##
##        E=207 * pow(10,9) #203#207
##        p=7.86 * 1000 #7.93#7.86
##        o=0.27
##        h=0.002
##
##        param = E * h * h * pi * pi / 3.0 / p / (1.0 - o * o)
##        c = pow(param * pow(freq,2),0.25)
##        time = sgn * (100.0 - dd) * 2.0 / 100.0 / c * 1000.0
##
##        plt.plot(time,corr,'+')
##        for i in range(len(freq)):
##            fr.write(str(int(freq[i])) + " " + str(time[i]) + " " + str(corr[i]) + "\r\n")

        count = count + 1
        if count > 50:
            break
    result.append(record)

##fr.close()
##plt.xlabel('theory dt/ms')
##plt.ylabel('cross correlation dt/ms')
##
##plt.xlim(-2,2)
##plt.ylim(-1.5,1.5)
##
##ax = plt.gca()
##box = ax.get_position()
##ax.set_position([box.x0, box.y0, box.width, box.height])
##plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
##plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)
##
##plt.show()


#result = np.transpose(result)
#for index,item in enumerate(result):
#    plt.plot(item,marker = markers_freq[index], label = str((axis_xf[index] + 1) * 20) + 'kHz')

for index,item in enumerate(result):
    plt.plot(item,marker = markers_freq[index], label = str((100 - (index + 4) * 10) * 2) + 'cm')

plt.xlabel('order')
plt.ylabel('dt/ms')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)

plt.show()

