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
fig = plt.figure(figsize=(5,3))

sgn = 1.0
interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval
start = 250000
end = 350000
result = []
pwd = os.getcwd()

result = []

savepathx = "rbf/train"
frx = open(savepathx, "w")

savepathc = "rbf/c"
frc = open(savepathc, "w")

freq_set = set([])

dd_range = np.arange(60,141,10)
#dd_range = [0,400,500,800,1700,2500,3100]

for dd in dd_range:
    num = 0
    record = []
    while True:
        num += 1
        if num > 20:
            break
        filepath = pwd + "//" + "input_data" + "//" + "pencil-train" + "//" + str(dd) + "//" + str(num)

        if os.path.isfile(filepath) == False:
            break
        
        with open(filepath, "rb") as fb:
            data = fb.read()

        ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
        ch1ch2 = np.array(ch1ch2)
        ch1ch2 = (ch1ch2-8192)*2.5/8192

        datay1 = ch1ch2[::2]
        datay2 = ch1ch2[1::2]

        datay1 = datay1[250000:350000]
        datay2 = datay2[250000:350000]
        count = len(datay1)

        fftnum = 10000
        std = 1500
        fftrepeat = 0

        axis_xf = range(int(fftnum/2))
        freq = [i * 10000000.0 / fftnum for i in axis_xf]

        index = 0
        magnitude1 = []
        magnitude2 = []
        while True:
            data1 = np.array(datay1[index:index + fftnum])
            data2 = np.array(datay2[index:index + fftnum])
            win = signal.gaussian(fftnum, std)
            data1 = np.multiply(data1,win)
            data2 = np.multiply(data2,win)
            data1 = np.abs(np.fft.fft(data1, fftnum)) / (fftnum / 2)
            data2 = np.abs(np.fft.fft(data2, fftnum)) / (fftnum / 2)
            magnitude1.append(data1)
            magnitude2.append(data2)
            index += int(fftnum * (1 - fftrepeat))
            if index + fftnum > count:
                break

        mag1 = sum(np.array(magnitude1)) / len(magnitude1)
        mag2 = sum(np.array(magnitude2)) / len(magnitude2)

        mag1 = mag1[20:40]
        maxmag1 = max(mag1)
        mag1 = mag1 / maxmag1

        mag2 = mag2[20:40]
        maxmag2 = max(mag2)
        mag2 = mag2 / maxmag2

        freq = np.array(freq[20:40])

        fa = []

        for i in range(20):
            if mag1[i] > 0.2 and mag2[i] > 0.2:
                fa.append(freq[i])

        datay1 = datay1[0:100000:5]
        datay2 = datay2[0:100000:5]

        wavelet = 'morl'
        c = pywt.central_frequency(wavelet)
        scales = np.array(float(c)) * fs / np.array(fa)

        [cfs1,frequencies1] = pywt.cwt(datay1,scales,wavelet,dt)
        [cfs2,frequencies2] = pywt.cwt(datay2,scales,wavelet,dt)
        power1 = abs(cfs1)
        power2 = abs(cfs2)

        for index,item in enumerate(fa):
            mean1 = power1[index].max()
            power1[index] = power1[index]/mean1
            mean2 = power2[index].max()
            power2[index] = power2[index]/mean2
            time1 = 0
            for i in range(1000,len(power1[index])):
               if power1[index][i + 1] > 0.2: 
                   time1 = i + 1
                   break
            time2 = 0
            for i in range(1000,len(power2[0]) - 2):
               if power2[index][i + 1] > 0.2: 
                   time2 = i + 1
                   break
            corr=(time1-time2) * dt * 1000
            print(dd,num,corr,item)
            if dd != 0 and corr != 0:
                speed = (100 - dd) * 2 / 100.0 / corr
                if speed > 0.01 and speed < 2:
                    freq_set.add(item)
                    frx.write("%.4f,%.3f\r\n"%(speed,item/100000.0))
            #fry.write("%.1f\r\n"%((100 - dd) * 2 / 100.0)))

frx.close()

freq_list = list(freq_set)
freq_list.sort()

for item in freq_list:
    frc.write("%.3f\r\n"%(item/1000000.0))

frc.close()

