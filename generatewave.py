import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import struct
from scipy import signal
import pywt
from math import sqrt

interval = 5
dt = 0.0000001 * interval
fs = 10000000 / interval
start = 250000
end = 450000
fig_size = 20

plt.rc('font',family='Times New Roman',size=10)

filepath = "/home/adoge/AE-location/input_data/test/100/1"

with open(filepath, "rb") as fb:
    data = fb.read()

ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
ch1ch2 = np.array(ch1ch2)
ch1ch2 = (ch1ch2-8192)*2.5/8192

datay1 = np.array(ch1ch2[::2])/3
datay2 = np.array(ch1ch2[1::2])/3
datay1 = datay1[240000:640000]
datay2 = datay2[240000:640000]

filepath = "/home/adoge/AE-location/input_data/sand/100/20"

with open(filepath, "rb") as fb:
    data = fb.read()

ch1ch2 = struct.unpack("<"+str(int(len(data)/2))+"H", data)
ch1ch2 = np.array(ch1ch2)
ch1ch2 = (ch1ch2-8192)*2.5/8192

datay3 = np.array(ch1ch2[::2])
datay4 = np.array(ch1ch2[1::2])
datax = np.array(range(len(datay3))) * 0.0000001

datay3[300000:700000] = datay3[300000:700000] + datay1
datay4[300000:700000] = datay4[300000:700000] + datay2

data1 = datay3[start:end:interval]
data2 = datay4[start:end:interval]
datax = np.array(range(len(data1))) * 0.0000001

fig = plt.figure(figsize=(5,3))

plt.plot(datax,data2,color='b',label = 'ch2')
plt.plot(datax,data1,color='g',label = 'ch1')
plt.xlabel('time/s')
plt.ylabel('amplitude/V')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
plt.legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=2)
plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.8)

plt.show()

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

with tf.Session() as sess:  
    
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    saver = tf.train.import_meta_graph('/home/adoge/AE-location/saver/SAE2/SAE.meta')
    saver.restore(sess,'/home/adoge/AE-location/saver/SAE2/SAE')  
    h = tf.get_collection('output_y')[0]
    y = tf.get_collection('output_y')[1]
    #l = tf.get_collection('output_y')[2]
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_x:0")
    
    hidden1,outputdata1 = sess.run([h,y],feed_dict ={x:[power1]})
    hidden2,outputdata2 = sess.run([h,y],feed_dict ={x:[power2]})
    
    power1 = np.reshape(power1,(20,fig_size))
    power2 = np.reshape(power2,(20,fig_size))
    outputdata1 = np.reshape(outputdata1,(20,fig_size))
    outputdata2 = np.reshape(outputdata2,(20,fig_size))
    #hidden1 = np.reshape(hidden1,(8,8))
    #hidden2 = np.reshape(hidden2,(8,8))
    
    
    plt.subplot(2,1,1)
    plt.imshow(power1,cmap=plt.get_cmap('rainbow'))
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.imshow(outputdata1,cmap=plt.get_cmap('rainbow'))
    plt.axis('off')
    plt.figure()
    plt.imshow(hidden1,cmap=plt.get_cmap('rainbow'))
    plt.axis('off')
    plt.show()
    
    plt.subplot(2,1,1)
    plt.imshow(power2,cmap=plt.get_cmap('rainbow'))
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.imshow(outputdata2,cmap=plt.get_cmap('rainbow'))
    plt.axis('off')
    plt.figure()
    plt.imshow(hidden2,cmap=plt.get_cmap('rainbow'))
    plt.axis('off')
    plt.show()
    
    loss1 = np.sum((power1 - outputdata1)**2)
    loss2 = np.sum((power2 - outputdata2)**2)
    print(loss1,loss2)
    
    coord.request_stop()
    coord.join(thread)
