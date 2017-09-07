# @Author: andreacasino
# @Date:   2017-08-06T23:09:42+01:00
# @Last modified by:   andreacasino
# @Last modified time: 2017-08-31T17:28:56+01:00
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
from settings import *
import numpy as np

def inference_reconstruction(keypoints_in):
    with tf.variable_scope('3D_Reconstruction'):

        layer_1 = layers.fully_connected(keypoints_in,KEYPOINTAMOUNT*2,scope='layer_1',activation_fn=None)
        layer_1 = tf.tanh(layer_1)

        layer_2 = layers.fully_connected(layer_1,KEYPOINTAMOUNT*2,scope='layer_2',activation_fn=None)
        layer_2 = tf.tanh(layer_2)

        layer_3 = layers.fully_connected(layer_2,KEYPOINTAMOUNT*2,scope='layer_3',activation_fn=None)
        layer_3 = tf.tanh(layer_3)

        layer_4 = layers.fully_connected(layer_3,KEYPOINTAMOUNT*2,scope='layer_4',activation_fn=None)
        layer_4 = tf.tanh(layer_4)

        layer_5 = layers.fully_connected(layer_4,KEYPOINTAMOUNT*2,scope='layer_5',activation_fn=None)
        layer_5 = tf.tanh(layer_5)

        layer_6 = layers.fully_connected(layer_5,KEYPOINTAMOUNT,scope='layer_6',activation_fn=None)
        layer_6 = tf.tanh(layer_6)

    return layer_6
