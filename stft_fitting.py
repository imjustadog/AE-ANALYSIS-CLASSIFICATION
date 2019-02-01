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

fftnum = 500#64#512#1024#8192
std = 125#16#128#256#1500

axis_xf = [11]
freq = np.array([fs * i / fftnum for i in axis_xf])

win = signal.gaussian(fftnum, std)
basis = []
for k in [11]:
    basis.append([complex(cos(2*pi/fftnum*k*n),sin(2*pi/fftnum*k*n)) for n in range(fftnum)])

basis = np.transpose(basis)

result = []

pwd = os.getcwd()

for dd in np.arange(42,163,10):
    count = 30
    record = []
    while True:
        filepath = pwd + "//" + "input_data" + "//" + "5" + "//" + str(dd) + "//" + str(count)

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
        print(corr)
        record.append(corr)

        count = count + 1
        if count > 50:
            break
    result.append(record)

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
        plt.plot(xi,xf-xi,'+')
plt.ylabel('localization error/cm')
print(np.mean(error))

plt.xlabel('distance difference/cm')
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)
plt.show()

