# @Author: andreacasino
# @Date:   2017-09-02T22:41:50+01:00
# @Last modified by:   andreacasino
# @Last modified time: 2017-09-07T16:13:15+01:00

import scipy.io as sio
import cv2
import numpy as np
from settings import *
import tensorflow as tf
import network

def normaliseKeypoints(x,y):

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    std_x = np.std(x)
    std_y = np.std(y)

    keypoints_xy = np.zeros([1,KEYPOINTAMOUNT*2])

    for kp in range(KEYPOINTAMOUNT):

        current_x = x[kp]
        current_y = y[kp]

        keypoints_xy[0,kp*2+0] = (current_x - mean_x)/((std_x+std_y)/2)
        keypoints_xy[0,kp*2+1] = (current_y - mean_y)/((std_x+std_y)/2)

    return keypoints_xy

def normaliseKeypointsIndividually(x,y):

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    std_x = np.std(x)
    std_y = np.std(y)

    keypoints_x = np.zeros([KEYPOINTAMOUNT])
    keypoints_y = np.zeros([KEYPOINTAMOUNT])

    for kp in range(KEYPOINTAMOUNT):

        current_x = x[kp]
        current_y = y[kp]

        keypoints_x[kp] = (current_x - mean_x)/((std_x+std_y)/2)
        keypoints_y[kp] = (current_y - mean_y)/((std_x+std_y)/2)

    return keypoints_x, keypoints_y

#Load heatmaps
loaded_heatmaps = np.load('./matlab/heatmaps.npy')
heatmap = loaded_heatmaps[:,:,0:10]

x_in = np.zeros([10])
y_in = np.zeros([10])

#Find keypoints
for kp in range(10):
    #Convert
    converted_heatmap = cv2.flip(heatmap[:,:,kp],0)


    #Find keypoints
    y,x = np.unravel_index(converted_heatmap.argmax(), converted_heatmap.shape)

    x_in[kp] = x
    y_in[kp] = y


#Normalise Keypoints
xy_in = normaliseKeypoints(x_in,y_in)
x_out, y_out = normaliseKeypointsIndividually(x_in,y_in)

tf.reset_default_graph()

with tf.variable_scope('Reconstruction'):
    keypoints_in = tf.placeholder(tf.float32, [1, KEYPOINTAMOUNT*2])
    keypoints_out = network.inference_reconstruction(keypoints_in)

init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, "./data_saves/modelSaves/model.ckpt")

    output = sess.run(keypoints_out,
                feed_dict={keypoints_in: xy_in})


np.save('z_out',output)
np.save('x_out',x_out)
np.save('y_out',y_out)
