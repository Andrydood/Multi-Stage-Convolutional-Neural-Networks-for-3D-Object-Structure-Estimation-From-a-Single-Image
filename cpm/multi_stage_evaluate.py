# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:36 2017

@author: Denis Tome'
"""

import sys
sys.path.insert(0, '../')

import tensorflow as tf
import cpm
import numpy as np
import random

from settings import *

def gaussianDistrib():

    batch_size = 1

    output = np.zeros([batch_size,HEIGHT,WIDTH,1])

    sigma = 25

    mean1=HEIGHT/2
    mean2=WIDTH/2

    for x1 in range(HEIGHT):
        for x2 in range(WIDTH):
            for batch in range(batch_size):
                output[batch,x1,x2,0] = np.exp(-((x1 - mean1)*(x1 - mean1))/(2*sigma*sigma)-((x2 - mean2)*(x2 - mean2))/(2*sigma*sigma))


    return output

imgIn = 4

epochLoad = 104573

tf.reset_default_graph()

gaussImg = gaussianDistrib()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

with tf.variable_scope('CPM'):
    pose_image_in = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])
    label_1st_lower = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH,KEYPOINTAMOUNT])
    heatmap_out = cpm.inference_pose_multi_stage(pose_image_in, label_1st_lower,gaussImg)

init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()
dir_training = 'mpi_v1'

#One for the 2nd stage layers and the other for the 1st stage
saver = tf.train.Saver()

img_batch = np.load("../data/testing_chair/images.npy")
heatmap_batch = np.load("../data/testing_chair/heatmaps.npy")

img_batch = img_batch[imgIn:imgIn+1,:,:,:]
heatmap_batch = heatmap_batch[imgIn:imgIn+1,:,:,:]

print("Training starting...")
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    saver.restore(sess, "./epoch_"+str(epochLoad)+"/model.ckpt")

    #Gather summaries
    hmOut= sess.run([ heatmap_out],
                            feed_dict={pose_image_in: img_batch})

    np.save("../data/output",hmOut)
    np.save("../data/input",img_batch)
    np.save("../data/ground_truth",heatmap_batch)
