import struct
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
from scipy import signal
from math import pi,sqrt
import tensorflow as tf
import pywt

import pywt
import struct

plt.figure(figsize=(6,3))
ax = plt.gca()

plt.rc('font',family='Times New Roman',size=10)
markers_freq = ['r+','b+']
curvename = ["positive samples", "negative samples"]

cmap = plt.get_cmap('rainbow') # this may fail on older versions of matplotlib
interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval
start = 250000
end = 350000
result = []

fig_size = 20


k0 = []
k0_norm = []


def calc_cos(k1,k2,k1_norm,k2_norm):
    max_cos1 = 0
    max_cos2 = 0
    for index in range(len(k0_norm)):
        k0k1 = (k0[index] * k1).sum()
        cos1 = k0k1 / k0_norm[index] / k1_norm
        k0k2 = (k0[index] * k2).sum()
        cos2 = k0k2 / k0_norm[index] / k2_norm
        loss1 = 1 - cos1
        cos_dis1 = 1 - loss1 * 10
        loss2 = 1 - cos2
        cos_dis2 = 1 - loss2 * 10
        if cos_dis1 > max_cos1:
            max_cos1 = cos_dis1
        if cos_dis2 > max_cos2:
            max_cos2 = cos_dis2
    return max_cos1,max_cos2

for dd in np.arange(100,101,10):
    filepath = "input_data" + "//" + "train1" + "//" + str(dd) + "//" + "0"

    if os.path.isfile(filepath) == False:
        break
        
    with open(filepath, "rb") as fb:
        data = fb.read()

    ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
    ch1ch2 = np.array(ch1ch2)
    ch1ch2 = (ch1ch2-8192)*2.5/8192

    datay1 = ch1ch2[::2]
    #datay2 = ch1ch2[1::2]
        
    data1 = datay1[start:end:interval]
    #data2 = datay2[start:end:interval]

    wavelet = 'morl'
    c = pywt.central_frequency(wavelet)
    fa = np.arange(400000, 20000 - 1, -20000)
    scales = np.array(float(c)) * fs / np.array(fa)

    [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
    #[cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
    power1 = (abs(cfs1)) ** 2
    #power2 = (abs(cfs2)) ** 2

    length_now = len(power1[0])
    power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
    #power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
    power1 = np.log10(np.mean(power1,axis=2))
    #power2 = np.log10(np.mean(power2,axis=2))

    mx = power1.max()
    mn = power1.min()
    power1 = (power1-mn) / (mx-mn) * 255.0
    power1 = np.floor(power1)

    #mx = power2.max()
    #mn = power2.min()
    #power2 = (power2-mn) / (mx-mn) * 255.0
    #power2 = np.floor(power2)

    k0.append(power1)
    k0_norm.append(sqrt((power1 ** 2).sum()))


record = []
for dd in [60,70,80,90,100,110,120,130,160]:
    count = 0
    while True:
        if count > 20:
            break

        filepath = "input_data" + "//" + "train" + "//" + str(dd) + "//" + str(count)

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
        fa = np.arange(400000, 20000 - 1, -20000)
        scales = np.array(float(c)) * fs / np.array(fa)

        [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
        [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
        power1 = (abs(cfs1)) ** 2
        power2 = (abs(cfs2)) ** 2

        length_now = len(power1[0])
        power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
        power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
        power1 = np.log10(np.mean(power1,axis=2))
        power2 = np.log10(np.mean(power2,axis=2))

        mx = power1.max()
        mn = power1.min()
        power1 = (power1-mn) / (mx-mn) * 255.0
        power1 = np.floor(power1)

        mx = power2.max()
        mn = power2.min()
        power2 = (power2-mn) / (mx-mn) * 255.0
        power2 = np.floor(power2)

        k1 = power1
        k2 = power2
        k1_norm = sqrt((k1 ** 2).sum())
        k2_norm = sqrt((k2 ** 2).sum())
 
        cos_dis1,cos_dis2=calc_cos(k1,k2,k1_norm,k2_norm)

        print(cos_dis1,cos_dis2)
        record.append(cos_dis1)
        record.append(cos_dis2)

        count = count + 1

result.append(record)

record = []
for dd in [1]:
    count = 0
    record = []
    while True:
        if count > 40:
            break

        filepath = "input_data" + "//" + "other" + "//" + "drop" + "//" + str(count)

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
        fa = np.arange(400000, 20000 - 1, -20000)
        scales = np.array(float(c)) * fs / np.array(fa)

        [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
        [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
        power1 = (abs(cfs1)) ** 2
        power2 = (abs(cfs2)) ** 2

        length_now = len(power1[0])
        power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
        power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
        power1 = np.log10(np.mean(power1,axis=2))
        power2 = np.log10(np.mean(power2,axis=2))

        mx = power1.max()
        mn = power1.min()
        power1 = (power1-mn) / (mx-mn) * 255.0
        power1 = np.floor(power1)

        mx = power2.max()
        mn = power2.min()
        power2 = (power2-mn) / (mx-mn) * 255.0
        power2 = np.floor(power2)

        k1 = power1
        k2 = power2
        k1_norm = sqrt((k1 ** 2).sum())
        k2_norm = sqrt((k2 ** 2).sum())

        cos_dis1,cos_dis2=calc_cos(k1,k2,k1_norm,k2_norm)

        print(cos_dis1,cos_dis2)
        record.append(cos_dis1)
        record.append(cos_dis2)

        count = count + 1
        
result.append(record)

for index in range(len(result)):
    record_x = []
    for y in range(len(result[index])):
        record_x.append(1 + np.random.uniform(-0.2,0.2))
    plt.plot(record_x,result[index],markers_freq[index],label = str(curvename[index]))





result = []
with tf.Session() as sess:  
    
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    saver = tf.train.import_meta_graph('saver/AE-cos/SAE.meta')
    saver.restore(sess,'saver/AE-cos/SAE')  
    y = tf.get_collection('output_y')[0]
    l = tf.get_collection('output_y')[1]
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_x:0")
    
    record = []
    for dd in np.arange(60,141,10):
        count = 0
        while True:
            if count > 20:
                break

            filepath = "input_data" + "//" + "train1" + "//" + str(dd) + "//" + str(count)

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
            fa = np.arange(400000, 20000 - 1, -20000)
            scales = np.array(float(c)) * fs / np.array(fa)

            [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
            [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
            power1 = (abs(cfs1)) ** 2
            power2 = (abs(cfs2)) ** 2

            length_now = len(power2[0])
            power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
            power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
            power1 = np.log10(np.mean(power1,axis=2))
            power2 = np.log10(np.mean(power2,axis=2))

            mx = power1.max()
            mn = power1.min()
            power1 = (power1-mn) / (mx-mn)
            power1 = power1.flatten()

            mx = power2.max()
            mn = power2.min()
            power2 = (power2-mn) / (mx-mn)
            power2 = power2.flatten()

            outputdata1,loss1 = sess.run([y,l],feed_dict ={x:[power1]})
            outputdata2,loss2 = sess.run([y,l],feed_dict ={x:[power2]})
            
            cos_dis1 = 1 - loss1 * 10
            cos_dis2 = 1 - loss2 * 10

            print(cos_dis1,cos_dis2)
            record.append(cos_dis1)
            record.append(cos_dis2)

            count = count + 1

    result.append(record)
    
    record = []
    for dd in [1]:
        count = 0
        while True:
            if count > 40:
                break

            filepath = "input_data" + "//" + "other" + "//" + "drop" + "//" + str(count)

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
            fa = np.arange(400000, 20000 - 1, -20000)
            scales = np.array(float(c)) * fs / np.array(fa)

            [cfs1,frequencies1] = pywt.cwt(data1,scales,wavelet,dt)
            [cfs2,frequencies2] = pywt.cwt(data2,scales,wavelet,dt)
            power1 = (abs(cfs1)) ** 2
            power2 = (abs(cfs2)) ** 2

            length_now = len(power2[0])
            power1 = np.reshape(power1,(len(power1),fig_size,int(length_now/fig_size)))
            power2 = np.reshape(power2,(len(power2),fig_size,int(length_now/fig_size)))
            power1 = np.log10(np.mean(power1,axis=2))
            power2 = np.log10(np.mean(power2,axis=2))

            mx = power1.max()
            mn = power1.min()
            power1 = (power1-mn) / (mx-mn)
            power1 = power1.flatten()

            mx = power2.max()
            mn = power2.min()
            power2 = (power2-mn) / (mx-mn)
            power2 = power2.flatten()

            outputdata1,loss1 = sess.run([y,l],feed_dict ={x:[power1]})
            outputdata2,loss2 = sess.run([y,l],feed_dict ={x:[power2]})
            
            cos_dis1 = 1 - loss1 * 10
            cos_dis2 = 1 - loss2 * 10

            print(cos_dis1,cos_dis2)
            record.append(cos_dis1)
            record.append(cos_dis2)

            count = count + 1

    result.append(record)
    
    
    coord.request_stop()
    coord.join(thread)

for index in range(len(result)):
    record_x = []
    for y in range(len(result[index])):
        record_x.append(2 + np.random.uniform(-0.2,0.2))
    plt.plot(record_x,result[index],markers_freq[index])

plt.xlim(0,3)
#plt.ylim(0,1)
xmajorLocator = MultipleLocator(1)
ax.xaxis.set_major_locator(xmajorLocator)
names = ["","Cosine Distance","Autoencoder",""]
plt.xticks([0,1,2,3],names)
plt.ylabel("similarity")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.7)

plt.show()
