
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io as sio
tf.set_random_seed(32)

types = 10

def fn(input):
    input = tf.reshape(input, [-1, 14 * 14 * 512])
    fc1 = slim.fully_connected(input, 2048, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_1')
    fc2 = slim.fully_connected(fc1, 1024, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_2')
    fc3 = slim.fully_connected(fc2, 512, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_3')
    fc4 = slim.fully_connected(fc3, 256, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_4')
    fc5 = slim.fully_connected(fc4, 128, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_5')
    fc6 = slim.fully_connected(fc5, 64, weights_regularizer=slim.l2_regularizer(0.05), scope='fc_6')
    pred = slim.fully_connected(fc6, types, activation_fn=tf.nn.softmax, scope='fc_7')
    return pred


def build_vgg19(input, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    vgg_rawnet = sio.loadmat('./data/imagenet-vgg-verydeep-19-SQ.mat')
    vgg_layers = vgg_rawnet['layers'][0]
    conv1_1 = build_net('conv', input, get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
    conv1_2 = build_net('conv', conv1_1, get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
    pool1 = build_net('pool', conv1_2)
    conv2_1 = build_net('conv', pool1, get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
    conv2_2 = build_net('conv', conv2_1, get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
    pool2 = build_net('pool', conv2_2)
    conv3_1 = build_net('conv', pool2, get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
    conv3_2 = build_net('conv', conv3_1, get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
    conv3_3 = build_net('conv', conv3_2, get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
    conv3_4 = build_net('conv', conv3_3, get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
    pool3 = build_net('pool', conv3_4)
    conv4_1 = build_net('conv', pool3, get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
    conv4_2 = build_net('conv', conv4_1, get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
    conv4_3 = build_net('conv', conv4_2, get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
    conv4_4 = build_net('conv', conv4_3, get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
    pool4 = build_net('pool', conv4_4)
    conv5_1 = build_net('conv', pool4, get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
    conv5_2 = build_net('conv', conv5_1, get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
    conv5_3 = build_net('conv', conv5_2, get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
    conv5_4 = build_net('conv', conv5_3, get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
    output = build_net('pool', conv5_4)
    return conv4_4


def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def load_data(FLAGS):

    if FLAGS.is_test == True:
        matfn = FLAGS.data
        data1 = sio.loadmat(matfn)
        data_test = data1['data_test'][:, 7:119, 7:119]
        return data_test

    elif FLAGS.mode == 'ZSL':
        matfn = './data/data.mat'
        data1 = sio.loadmat(matfn)
        data = data1['data'][:, 7:119, 7:119]
        data_seg = data1['data_seg'][:, 7:119, 7:119]
        data = data * data_seg
        data_simu = data1['data_simu'][:, 7:119, 7:119]
        data_simu_seg = data1['data_simu_seg'][:, 7:119, 7:119]
        data_simu = data_simu * data_simu_seg
        T72_15 = data1['T72_15'][:, 7:119, 7:119]
        T72_15_seg = data1['T72_15_seg'][:, 7:119, 7:119]
        T72_15 = T72_15 * T72_15_seg
    elif FLAGS.mode == '-FANS':
        matfn = './data/data_fans.mat'
        data1 = sio.loadmat(matfn)
        data = data1['data'][:, 7:119, 7:119]
        data_seg = data1['data_seg'][:, 7:119, 7:119]
        data = data * data_seg
        data_simu = data1['data_simu'][:, 7:119, 7:119]
        data_simu_seg = data1['data_simu_seg'][:, 7:119, 7:119]
        data_simu = data_simu * data_simu_seg
        T72_15 = data1['T72_15'][:, 7:119, 7:119]
        T72_15_seg = data1['T72_15_seg'][:, 7:119, 7:119]
        T72_15 = T72_15 * T72_15_seg
    elif FLAGS.mode == '-Style':
        matfn = './data/data_style.mat'
        data1 = sio.loadmat(matfn)
        data = data1['data'][:, 7:119, 7:119]
        data_seg = data1['data_seg'][:, 7:119, 7:119]
        data = data * data_seg
        data_simu = data1['data_simu'][:, 7:119, 7:119]
        data_simu_seg = data1['data_simu_seg'][:, 7:119, 7:119]
        data_simu = data_simu * data_simu_seg
        T72_15 = data1['T72_15'][:, 7:119, 7:119]
        T72_15_seg = data1['T72_15_seg'][:, 7:119, 7:119]
        T72_15 = T72_15 * T72_15_seg
    elif FLAGS.mode == '-Segmentation':
        matfn = './data/data.mat'
        data1 = sio.loadmat(matfn)
        data = data1['data'][:, 7:119, 7:119]
        data_simu = data1['data_simu'][:, 7:119, 7:119]
        T72_15 = data1['T72_15'][:, 7:119, 7:119]
    else:
        raise Exception("Invalid mode!", FLAGS.mode)

    l_train = data1['y_']
    label = np.zeros([len(data), types])
    label[:, 0:types] = l_train

    np.random.seed(1)
    idx = np.random.permutation(len(data))
    data_train = data[idx[0:int(len(data) * 0.8)]]
    label_train = label[idx[0:int(len(label) * 0.8)]]
    data_test = data[idx[int(len(data) * 0.8):]]
    label_test = label[idx[int(len(label) * 0.8):]]
    train_size = len(data_train)
    test_size = len(data_test)

    data_train = np.concatenate((data_train, data_simu), axis=0)
    label_train = np.concatenate((label_train, np.tile([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], (359, 1))), axis=0)

    data_test = np.concatenate((data_test, T72_15), axis=0)
    label_test = np.concatenate((label_test, np.tile([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], (196, 1))), axis=0)

    return data_train, label_train, data_test, label_test, train_size, test_size

