import struct
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rc('font',family='Times New Roman',size=10)

pwd = os.getcwd()
count = 0
while True:
    fb = open(pwd + "//" + "input_data" + "//" + "train1" + "//" + "60" + "//" + str(count), "rb")
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

    fig = plt.figure()
    plt.plot(datax,datay2,color='b')
    plt.plot(datax,datay1,color='g')
    
    
    plt.xlabel('time/s')
    plt.ylabel('amplitude/V')
    plt.title(str(count))
    plt.show()
    count = count + 1
