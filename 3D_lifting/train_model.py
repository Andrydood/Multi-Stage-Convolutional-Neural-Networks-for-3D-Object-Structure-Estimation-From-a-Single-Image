# @Author: andreacasino
# @Date:   2017-08-30T20:00:34+01:00
# @Last modified by:   andreacasino
# @Last modified time: 2017-09-05T16:36:07+01:00



import tensorflow as tf
import network
import numpy as np
import random
import scipy.io as sio

from settings import *

image_amount = 178695


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


def importKeypoints():

    keypoints_xy = np.zeros([image_amount,KEYPOINTAMOUNT*2])
    keypoints_z = np.zeros([image_amount,KEYPOINTAMOUNT])

    keypoints_x = np.zeros([image_amount,KEYPOINTAMOUNT])
    keypoints_y = np.zeros([image_amount,KEYPOINTAMOUNT])

    #Load keypoints
    keypoints_mat = sio.loadmat('./keypoints.mat')
    x_in = keypoints_mat['x_in']
    y_in = keypoints_mat['y_in']
    z_out = keypoints_mat['z_out']

    for img in range(image_amount):
        current_x = x_in[img,:]
        current_y = y_in[img,:]
        current_z = z_out[img,:]

        keypoints_xy[img,:] = normaliseKeypoints(current_x,current_y)
        keypoints_z[img,:] = z_out[img,:]

        keypoints_x[img,:] = current_x
        keypoints_y[img,:] = current_y

    return keypoints_xy, keypoints_z, keypoints_x,keypoints_y


tf.reset_default_graph()

with tf.variable_scope('Reconstruction'):
    keypoints_in = tf.placeholder(tf.float32, [None, KEYPOINTAMOUNT*2])
    keypoints_gt = tf.placeholder(tf.float32, [None, KEYPOINTAMOUNT])
    keypoints_out = network.inference_reconstruction(keypoints_in)
with tf.name_scope('Loss'):
    total_loss = tf.losses.mean_squared_error(keypoints_out, keypoints_gt)

with tf.name_scope('Optimiser'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate= LEARNING_RATE ).minimize(total_loss)

init = tf.global_variables_initializer()

summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

keypoints_xy, keypoints_z, keypoints_x, keypoints_y = importKeypoints()

print("Training starting...")
with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter("./data_saves/logs",sess.graph)

    for epoch in range(1000000000):

        botRange = 0

        for batch in range(50,178100,50):

            topRange = batch

            current_xy = keypoints_xy[botRange:topRange,:]
            current_z = keypoints_z[botRange:topRange,:]

            botRange = topRange

            #Gather summaries
            sess.run(train_op,
                        feed_dict={keypoints_in: current_xy,
                                  keypoints_gt: current_z})


        test_xy = keypoints_xy[178100:178600,:]
        test_z = keypoints_z[178100:178600,:]

        loss,z_out = sess.run([total_loss,keypoints_out],
                                feed_dict={keypoints_in: test_xy,
                                          keypoints_gt: test_z})

        errMean = (np.mean(np.absolute(z_out - test_z)));
        errStd = (np.std(np.absolute(z_out - test_z)));

        print("Epoch: "+str(epoch)+" Loss: "+str(loss)+" Training Error Mean: "
                        +str(errMean)+"Training Error Std: "+str(errStd))


        #Save model
        save_path = saver.save(sess, "./data_saves/modelSaves/model.ckpt")
        np.save("x",keypoints_x[178100:178600,:])
        np.save("y",keypoints_y[178100:178600,:])
        np.save("z_r",keypoints_z[178100:178600,:])
        np.save("z",z_out)
