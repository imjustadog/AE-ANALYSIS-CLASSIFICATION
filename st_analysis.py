import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi
from stockwell import st
import heapq

import pywt
import struct

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_',','] 
fig = plt.figure(figsize=(5,3))

sgn = 1.0
interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval

result = []
pwd = os.getcwd()

result = []

#savepath = pwd + "//" + "wt_1.txt"
#fr = open(savepath, "w")
curvename = []

for dd in np.arange(60,101,10):
    curvename.append(dd)
    count = 0
    record = []
    while True:
        filepath = pwd + "//" + "input_data" + "//" + "train_knock" + "//" +  str(dd) + "//" + str(count)

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

        data1 = data1[int(250000 / interval):int(350000 / interval)]
        data2 = data2[int(250000 / interval):int(350000 / interval)]

        fa = np.arange(20000, 60000 + 1, 10000)
        cfs1 = []
        cfs2 = []
        for a in fa:
            fmaxmin = int(a*(len(data1)*dt))
            cfs1.append(st.st(data1, fmaxmin, fmaxmin)[0])
            cfs2.append(st.st(data2, fmaxmin, fmaxmin)[0])

        power1 = (abs(np.array(cfs1)))
        power2 = (abs(np.array(cfs2)))

        corr_time = []
        corr_value = []
        corr_frequency = []
        for i in range(len(power1)):
            mean1 = power1[i].mean()
            power1[i] = power1[i] / mean1
            mean2 =  power2[i].mean()
            power2[i] = power2[i] / mean2
            temp = signal.correlate(power1[i],power2[i], mode='same',method='fft')
            corr_frequency.append(i*10 + 20)
            max2=heapq.nlargest(2,temp)
            corr_value.append(max2[0]/max2[1])
            corr_time.append((np.where(temp == max(temp))[0][0]-len(temp) / 2 ) * dt * 1000)

        corr_index = np.where(corr_value == max(corr_value))[0][0]
        print(dd,int(corr_frequency[corr_index]),corr_time[corr_index])
        record.append(corr_time[corr_index])
        
##        E=207 * pow(10,9) #203#207
##        p=7.86 * 1000 #7.93#7.86
##        o=0.27
##        h=0.002
##
##        freq = np.array(frequencies1)
##        param = E * h * h * pi * pi / 3.0 / p / (1.0 - o * o)
##        c = pow(param * pow(freq,2),0.25)
##        time = sgn * (100.0 - dd) * 2.0 / 100.0 / c * 1000.0
##
##        plt.plot(time,corr,'+')
##        for i in range(len(freq)):
##            fr.write(str(int(freq[i])) + " " + str(time[i]) + " " + str(corr[i]) + "\r\n")

        count = count + 1
        if count > 20:
            break
    result.append(record)

#fr.close()

#result = np.transpose(result)
#for index,item in enumerate(result):
#    plt.plot(item,marker = markers_freq[index], label = str((axis_xf[index] + 1) * 20) + 'kHz')


for index,item in enumerate(result):
    plt.plot(item,marker = markers_freq[index], label = str(curvename[index]) + 'cm')

plt.xlabel('order')
plt.ylabel('dt/ms')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)

plt.show()
