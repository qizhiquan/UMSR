#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import time

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py


def UMSR_g(t_image, is_train=False, reuse=False):
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')  #(batch_size, 112, 112, 64)
        conv1 = network
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        conv2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')  #(batch_size, 56, 56, 128)

        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        conv3 = network
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')  #(batch_size, 28, 28, 256)

        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        conv4 = network
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  #(batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        conv5 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)

        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv1, conv2, conv3, conv4, conv5

def gram_matrix(features):
    shape = tf.shape(features)
    feature_reshaped = tf.reshape(features, [shape[0], shape[1]*shape[2], shape[3]])
    gram = tf.matmul(tf.transpose(feature_reshaped, [0,2,1]), feature_reshaped)

    gram /= tf.cast((shape[3] * shape[2] * shape[1]), tf.float32)
    return gram


def gram_scale_loss1(features_predict,features_target):


    featrues_predict_resize1 = tf.image.resize_images(features_predict,[56,56],method=0)
    featrues_predict_resize2 = tf.image.resize_images(features_predict,[28,28],method=0)
    featrues_predict_resize3 = tf.image.resize_images(features_predict,[14,14],method=0)
    featrues_predict_resize4 = tf.image.resize_images(features_predict,[7,7],method=0)

    featrues_target_resize1 = tf.image.resize_images(features_target,[56,56],method=0)
    featrues_target_resize2 = tf.image.resize_images(features_target,[28,28],method=0)
    featrues_target_resize3 = tf.image.resize_images(features_target,[14,14],method=0)
    featrues_target_resize4 = tf.image.resize_images(features_target,[7,7],method=0)

    loss0 = tl.cost.mean_squared_error(gram_matrix(features_predict), gram_matrix(features_target), is_mean=True)
    loss1 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize1), gram_matrix(featrues_target_resize1), is_mean=True)
    loss2 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize2), gram_matrix(featrues_target_resize2), is_mean=True)
    loss3 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize3), gram_matrix(featrues_target_resize3), is_mean=True)
    loss4 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize4), gram_matrix(featrues_target_resize4), is_mean=True)

    gram_scale_loss1 = (loss0 + loss1 + loss2 + loss3 + loss4) / 5

    return  gram_scale_loss1


def gram_scale_loss2(features_predict,features_target):


    featrues_predict_resize1 = tf.image.resize_images(features_predict,[28,28],method=0)
    featrues_predict_resize2 = tf.image.resize_images(features_predict,[14,14],method=0)
    featrues_predict_resize3 = tf.image.resize_images(features_predict,[7,7],method=0)
    featrues_predict_resize4 = tf.image.resize_images(features_predict,[3,3],method=0)

    featrues_target_resize1 = tf.image.resize_images(features_target,[28,28],method=0)
    featrues_target_resize2 = tf.image.resize_images(features_target,[14,14],method=0)
    featrues_target_resize3 = tf.image.resize_images(features_target,[7,7],method=0)
    featrues_target_resize4 = tf.image.resize_images(features_target,[3,3],method=0)

    loss0 = tl.cost.mean_squared_error(gram_matrix(features_predict), gram_matrix(features_target), is_mean=True)
    loss1 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize1), gram_matrix(featrues_target_resize1), is_mean=True)
    loss2 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize2), gram_matrix(featrues_target_resize2), is_mean=True)
    loss3 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize3), gram_matrix(featrues_target_resize3), is_mean=True)
    loss4 = tl.cost.mean_squared_error(gram_matrix(featrues_predict_resize4), gram_matrix(featrues_target_resize4), is_mean=True)

    gram_scale_loss2 = (loss0 + loss1 + loss2 + loss3 + loss4) / 5

    return  gram_scale_loss2
