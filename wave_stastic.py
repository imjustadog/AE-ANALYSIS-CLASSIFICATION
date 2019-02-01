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
        
        count = count + 1
        if count > 20:
            break
    result.append(record)

#fr.close()

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
