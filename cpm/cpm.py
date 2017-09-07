# @Author: andreacasino
# @Date:   2017-08-06T23:09:42+01:00
# @Last modified by:   andreacasino
# @Last modified time: 2017-09-06T16:41:20+01:00
import sys
sys.path.insert(0, '../')

import tensorflow as tf
import tensorflow.contrib.layers as layers
from settings import *
import numpy as np

def inference_pose_stage_1(image, label_1st_lower):
    with tf.variable_scope('PoseNet'):
        slim = tf.contrib.slim

        image = tf.image.convert_image_dtype(image,tf.float32)
        label_1st_lower = tf.image.convert_image_dtype(label_1st_lower,tf.float32)

        image = tf.subtract(tf.divide(image,255),0.5)

        conv1_1 = layers.conv2d(image, 64, 3, 1, activation_fn=None, scope='conv1_1' )
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2')
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1, activation_fn=None, scope='conv2_1')
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = layers.conv2d(conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2')
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1, activation_fn=None, scope='conv3_1' )
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = layers.conv2d(conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2')
        conv3_2 = tf.nn.relu(conv3_2)
        conv3_3 = layers.conv2d(conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3' )
        conv3_3 = tf.nn.relu(conv3_3)
        conv3_4 = layers.conv2d(conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4')
        conv3_4 = tf.nn.relu(conv3_4)
        pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1, activation_fn=None, scope='conv4_1')
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2' )
        conv4_2 = tf.nn.relu(conv4_2)
        conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
        conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
        conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1, activation_fn=None, scope='conv4_4_CPM')
        conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
        conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1, activation_fn=None, scope='conv4_5_CPM')
        conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
        conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1, activation_fn=None, scope='conv4_6_CPM')
        conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
        conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1, activation_fn=None, scope='conv4_7_CPM')
        conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
        conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1, activation_fn=None, scope='conv5_1_CPM')
        conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
        conv5_2_CPM = layers.conv2d(conv5_1_CPM, KEYPOINTAMOUNT, 1, 1, activation_fn=None, scope='conv5_2_CPM')

        #Reshape output to initial image size
        label_1st_lower_reshaped = tf.image.resize_images(label_1st_lower,[int(HEIGHT/8),int(WIDTH/8)])

        tf.losses.mean_squared_error(conv5_2_CPM, label_1st_lower_reshaped)

        conv5_2_CPM = tf.image.resize_images(conv5_2_CPM,[int(HEIGHT),int(WIDTH)])

        # tf.summary.image("Input Image 1 Label 1",label_1st_lower[4:5,:,:,0:1])
        # tf.summary.image("Input Image 1 Label 2",label_1st_lower[4:5,:,:,2:3])
        # tf.summary.image("Input Image 1 Label 3",label_1st_lower[4:5,:,:,4:5])
        tf.summary.image("Image",image[4:5,:,:,:])
        tf.summary.image("Output Label 1",conv5_2_CPM[4:5,:,:,0:1])
        tf.summary.image("Output Label 2",conv5_2_CPM[4:5,:,:,1:2])
        tf.summary.image("Output Label 3",conv5_2_CPM[4:5,:,:,2:3])
        tf.summary.image("Output Label 4",conv5_2_CPM[4:5,:,:,3:4])
        tf.summary.image("Output Label 5",conv5_2_CPM[4:5,:,:,4:5])
        tf.summary.image("Output Label 6",conv5_2_CPM[4:5,:,:,5:6])
        tf.summary.image("Output Label 7",conv5_2_CPM[4:5,:,:,6:7])
        tf.summary.image("Output Label 8",conv5_2_CPM[4:5,:,:,7:8])
        tf.summary.image("Output Label 9",conv5_2_CPM[4:5,:,:,8:9])
        tf.summary.image("Output Label 10",conv5_2_CPM[4:5,:,:,9:10])
        tf.summary.image("Output Label 11",conv5_2_CPM[4:5,:,:,10:11])


    return conv5_2_CPM


def inference_pose_multi_stage(image, label_1st_lower,gaussImg):
    with tf.variable_scope('PoseNet'):


        label_1st_lower_reshaped = tf.image.resize_images(label_1st_lower,[int(HEIGHT/8),int(WIDTH/8)])
        gaussImg = tf.image.resize_images(gaussImg,[int(HEIGHT/8),int(WIDTH/8)])

        image = tf.image.convert_image_dtype(image,tf.float32)
        label_1st_lower = tf.image.convert_image_dtype(label_1st_lower,tf.float32)

        image = tf.subtract(tf.divide(image,255),0.5)

        conv1_1 = layers.conv2d(image, 64, 3, 1, activation_fn=None, scope='conv1_1'  )
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2'  )
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1, activation_fn=None, scope='conv2_1'  )
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = layers.conv2d(conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2'  )
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1, activation_fn=None, scope='conv3_1'  )
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = layers.conv2d(conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2'  )
        conv3_2 = tf.nn.relu(conv3_2)
        conv3_3 = layers.conv2d(conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3'  )
        conv3_3 = tf.nn.relu(conv3_3)
        conv3_4 = layers.conv2d(conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4'  )
        conv3_4 = tf.nn.relu(conv3_4)
        pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1, activation_fn=None, scope='conv4_1'  )
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2'  )
        conv4_2 = tf.nn.relu(conv4_2)

        conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM' )
        conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
        conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1, activation_fn=None, scope='conv4_4_CPM' )
        conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
        conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1, activation_fn=None, scope='conv4_5_CPM' )
        conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
        conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1, activation_fn=None, scope='conv4_6_CPM' )
        conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
        conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1, activation_fn=None, scope='conv4_7_CPM' )
        conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
        conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1, activation_fn=None, scope='conv5_1_CPM' )
        conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
        conv5_2_CPM = layers.conv2d(conv5_1_CPM, KEYPOINTAMOUNT, 1, 1, activation_fn=None, scope='conv5_2_CPM' )

        #Reshape output to initial image size
        #Concatenate image and output

        tf.losses.mean_squared_error(conv5_2_CPM, label_1st_lower_reshaped)

        # loss defined
        concat_stage2 = tf.concat([conv5_2_CPM, conv4_7_CPM, gaussImg], 3)
        Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1, activation_fn=None, scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
        Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1, activation_fn=None, scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
        Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1, activation_fn=None, scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
        Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1, activation_fn=None, scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
        Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1, activation_fn=None, scope='Mconv5_stage2')
        Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
        Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1, activation_fn=None, scope='Mconv6_stage2')
        Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
        Mconv7_stage2 = layers.conv2d(Mconv6_stage2, KEYPOINTAMOUNT, 1, 1, activation_fn=None, scope='Mconv7_stage2')

        #defining loss
        tf.losses.mean_squared_error(Mconv7_stage2, label_1st_lower_reshaped)

        Mconv7_stage2 = tf.image.resize_images(Mconv7_stage2,[int(HEIGHT),int(WIDTH)])

        tf.summary.image("Image",image[4:5,:,:,:])
        tf.summary.image("Output Label 1",Mconv7_stage2[4:5,:,:,0:1])
        tf.summary.image("Output Label 2",Mconv7_stage2[4:5,:,:,1:2])
        tf.summary.image("Output Label 3",Mconv7_stage2[4:5,:,:,2:3])
        tf.summary.image("Output Label 4",Mconv7_stage2[4:5,:,:,3:4])
        tf.summary.image("Output Label 5",Mconv7_stage2[4:5,:,:,4:5])
        tf.summary.image("Output Label 6",Mconv7_stage2[4:5,:,:,5:6])
        tf.summary.image("Output Label 7",Mconv7_stage2[4:5,:,:,6:7])
        tf.summary.image("Output Label 8",Mconv7_stage2[4:5,:,:,7:8])
        tf.summary.image("Output Label 9",Mconv7_stage2[4:5,:,:,8:9])
        tf.summary.image("Output Label 10",Mconv7_stage2[4:5,:,:,9:10])
        tf.summary.image("Output Label 11",Mconv7_stage2[4:5,:,:,10:11])
        tf.summary.image("Gauss",gaussImg)



    return Mconv7_stage2
