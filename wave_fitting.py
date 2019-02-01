import struct
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from math import pi

import pywt
import struct

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['.','o','v','1','2','3','4','^','<','>','s','p','*','h','H','+','x','D','d','|','_',','] 
fig = plt.figure(figsize=(5,5))

sgn = 1.0
interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval

result = []
pwd = os.getcwd()

result = []

#savepath = pwd + "//" + "wt_1.txt"
#fr = open(savepath, "w")

for dd in np.arange(42,163,10):
    count = 0
    record = []
    while True:
        filepath = pwd + "//" + "input_data" + "//" + "4" + "//" + str(dd) + "//" + str(count)

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

        temp = signal.correlate(data1,data2, mode='same',method='fft')
        corr=(np.where(temp == max(temp))[0][0]-len(temp) / 2 ) * dt * 1000
        print(corr)
        record.append(corr)
        
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


plt.subplot(2,1,1)
x = []
y = []
for index,item in enumerate(result):
    for yi in item:
        xi = 120 - index * 20
        x.append(xi)
        y.append(yi)
        plt.plot(xi, yi, '+')

f = np.polyfit(x,y,1)
xf = [120 - i * 20 for i in range(0,13)]
yf = [xi * f[0] + f[1] for xi in xf]
plt.plot(xf,yf,'r-')
plt.ylabel('time difference/ms')
print(f)

plt.subplot(2,1,2)
error = []
for index,item in enumerate(result):
    for yi in item:
        xi = 120 - index * 20
        xf = (yi - f[1])/f[0]
        error.append(abs(xf - xi))
        plt.plot(xi,xf,'+')
plt.ylabel('calculation result/cm')
print(np.mean(error))

plt.xlabel('distance difference/cm')
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()
