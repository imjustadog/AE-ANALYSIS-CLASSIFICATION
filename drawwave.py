import struct
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rc('font',family='Times New Roman',size=10)

pwd = os.getcwd()
fb = open(pwd + "//" + "input_data" +"//" + "other" + "//" + "drop" + "//" + "3", "rb")
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
    datax.append(x * 0.0000001)
    datay1.append(ch1)
    datay2.append(ch2)

fig = plt.figure(figsize=(5,3))

plt.plot(datax,datay2,color='b',label = 'ch2')
plt.plot(datax,datay1,color='g',label = 'ch1')
plt.xlabel('time/s')
plt.ylabel('amplitude/V')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)

plt.show()
