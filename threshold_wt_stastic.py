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

#savepath = pwd + "//" + "wt_1.txt"
#fr = open(savepath, "w")

dd_range = np.arange(60,61,10)

for dd in dd_range:
    count = 1
    record = []
    while True:
        filepath = pwd + "//" + "input_data" + "//" + "train1" + "//" + str(dd) + "//" + str(count)

        if os.path.isfile(filepath) == False:
            break
        
        with open(filepath, "rb") as fb:
            data = fb.read()

        ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
        ch1ch2 = np.array(ch1ch2)
        ch1ch2 = (ch1ch2-8192)*2.5/8192

        datay1 = ch1ch2[::2]
        datay2 = ch1ch2[1::2]

        data1 = datay1[start:end:interval]
        data2 = datay2[start:end:interval]

        wavelet = 'morl'
        c = pywt.central_frequency(wavelet)
        fa = [161000] #np.arange(400000, 200000 - 1, -20000)
        scales = np.array(float(c)) * fs / np.array(fa)

        [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
        [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
        power1 = abs(cfs1)
        power2 = abs(cfs2)

        mean1 = power1[0].max()
        power1[0] = power1[0]/mean1
        mean2 = power2[0].max()
        power2[0] = power2[0]/mean2

        time1 = 0
        for i in range(1000,len(power1[0]) - 2):
           #if power1[0][i + 1] >= power1[0][i] and power1[0][i + 1] > power1[0][i + 2]:
               if power1[0][i + 1] > 0.2: 
                   time1 = i + 1
                   break
  
        time2 = 0
        for i in range(1000,len(power2[0]) - 2):
           #if power2[0][i + 1] >= power2[0][i] and power2[0][i + 1] > power2[0][i + 2]:
               if power2[0][i + 1] > 0.2: #and power2[0][i + 1] > 10 * max2:
                   time2 = i + 1
                   break

        corr=(time1-time2) * dt * 1000
        print(corr)
        #print(power1[0][time1], time1, power2[0][time2], time2, corr)
        record.append(corr)
        
        count = count + 1
        if count > 20:
            break
    result.append(record)

#fr.close()

#result = np.transpose(result)
#for index,item in enumerate(result):
#    plt.plot(item,marker = markers_freq[index], label = str((axis_xf[index] + 1) * 20) + 'kHz')


for index,item in enumerate(result):
    plt.plot(item,marker = markers_freq[index], label = str(dd_range[index]) + 'cm')

plt.xlabel('order')
plt.ylabel('dt/ms')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)

plt.show()
