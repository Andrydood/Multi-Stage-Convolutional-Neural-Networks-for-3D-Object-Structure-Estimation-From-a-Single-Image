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


tf.reset_default_graph()

with tf.variable_scope('CPM'):
    pose_image_in = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])
    label_1st_lower = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH,KEYPOINTAMOUNT])
    heatmap_out = cpm.inference_pose_stage_1(pose_image_in, label_1st_lower)
with tf.name_scope('Loss'):
    total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
    all_losses = tf.losses.get_losses()

    # summary
    with tf.variable_scope('Losses'):
        for i in range(len(all_losses)):
            tf.summary.scalar("stage %r" % i, all_losses[i])
        tf.summary.scalar("TOTAL", total_loss)

with tf.name_scope("Accuracy"):
        #To test the network, find the maximum value in every ideal heatmap in both axes
        #then find the difference between their location.
        #Then add the axis differences, then find the mean difference
        maxX_gt = tf.argmax(tf.reduce_max(label_1st_lower[:,:,:,0:10],1),1)
        maxY_gt = tf.argmax(tf.reduce_max(label_1st_lower[:,:,:,0:10],2),1)

        maxX_out = tf.argmax(tf.reduce_max(heatmap_out[:,:,:,0:10],1),1)
        maxY_out = tf.argmax(tf.reduce_max(heatmap_out[:,:,:,0:10],2),1)

        Xdiff = tf.abs(tf.subtract(maxX_gt,maxX_out))
        Ydiff = tf.abs(tf.subtract(maxY_gt,maxY_out))

        totalDiff = tf.sqrt(tf.cast(tf.add(tf.square(Xdiff),tf.square(Ydiff)),tf.float32))

        difference = tf.reduce_mean(totalDiff)

        tf.summary.scalar("Distance",difference)

with tf.name_scope('Optimiser'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate= LEARNING_RATE ).minimize(total_loss)

# Don't restore last variable
restore_var = [v for v in tf.global_variables() if 'conv5_2_CPM' not in v.name and
                                                    'conv1_1' not in v.name]

init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()
dir_training = 'mpi_v1'

loader = tf.train.Saver(var_list=restore_var)
saver = tf.train.Saver()

#Load test data
testImages = np.load("../data/testing_chair/images.npy")
testHeatmaps = np.load("../data/testing_chair/heatmaps.npy")

print(testImages.shape)

print("Training starting...")
with tf.Session() as sess:
    sess.run(init)

    testWriter = tf.summary.FileWriter("./dataNew/logs/test",sess.graph)
    trainWriter = tf.summary.FileWriter("./dataNew/logs/train",sess.graph)

    loader.restore(sess, "../data/init_session/init")

    for epoch in range(1000):


        for imageIdx, imageSet in enumerate(random.sample(range(TRAININGIMAGEAMOUNT), TRAININGIMAGEAMOUNT)):

                currentImages = np.load("../data/training_chair/images_"+str(imageSet)+".npy")
                currentHeatmaps = np.load("../data/training_chair/heatmaps_"+str(imageSet)+".npy")

                botRange = 0

                for idx, num in enumerate(range(10,101,10)):

                    step = epoch*TRAININGIMAGEAMOUNT*10+imageIdx*10+idx

                    topRange = num

                    #Gathering images to be used for training
                    img_batch = currentImages[botRange:topRange,:,:,:]
                    label_batch= currentHeatmaps[botRange:topRange,:,:,:]

                    botRange = topRange

                    # Minimise loss
                    sess.run(train_op,feed_dict={pose_image_in: img_batch,
                                                  label_1st_lower: label_batch})

                    #Gather summaries
                    loss,hmOut,summary, diff = sess.run([all_losses, heatmap_out,summary_op,difference],
                                            feed_dict={pose_image_in: img_batch,
                                                        label_1st_lower: label_batch})

                    print("Step: "+str(step)+" Loss: "+str(loss)+" Distance: "+str(diff))

                    trainWriter.add_summary(summary, step)

                #Random sample of 20 test images
                testIdx = random.randint(0,200)
                imageInputTest = testImages[testIdx:testIdx+20,:,:,:]
                heatmapInputTest = testHeatmaps[testIdx:testIdx+20,:,:,:]

                #Test data every 100 images
                loss,hmOut,summary,diff = sess.run([all_losses, heatmap_out,summary_op,difference],
                                                feed_dict={pose_image_in: imageInputTest,
                                                label_1st_lower: heatmapInputTest})

                testWriter.add_summary(summary, step)

        #Save model
        save_path = saver.save(sess, "./dataNew/modelSaves/epoch_"+str(epoch)+"/model.ckpt")
