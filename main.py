# -*- coding: utf-8 -*-
"""
EM Simulation-Aided Zero-Shot Learning for SAR Automatic Target Recognition
@author: Qian Song
"""

from __future__ import division
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
from utils import *
import os
np.random.seed(0)
tf.set_random_seed(32)

flags = tf.app.flags
flags.DEFINE_boolean('is_test', False, "Flag of opertion: True is for training")
flags.DEFINE_string('data', './data/test_165.mat', "Path of test image")
flags.DEFINE_string('dir', './checkpoint', "Path of saving model")
flags.DEFINE_string('result_dir', './result', "Path of saving results")
flags.DEFINE_string('mode', 'ZSL', "Training mode, include ZSL, -FANS -Style, and -Segmentation. ")
FLAGS = flags.FLAGS
st = time.time()

if not os.path.exists(FLAGS.dir + '/' + FLAGS.mode):
    os.makedirs(FLAGS.dir + '/' + FLAGS.mode)
if not os.path.exists(FLAGS.result_dir + '/' + FLAGS.mode):
    os.makedirs(FLAGS.result_dir + '/' + FLAGS.mode)


class zsl_model(object):
    def __init__(self, sess):
        self.sess = sess
        self.learning_rate = 1e-5
        self.types = 10
        self.img_size = 112
        self.Batch_Size = 50
        self.Epoches = 15
        self.build_model()

    def build_model(self):
        with tf.variable_scope(tf.get_variable_scope()):
            self.label = tf.placeholder(tf.float32, [None, self.types])
            self.input_image = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
            self.vgg_output = build_vgg19(self.input_image)
            self.predict = fn(self.vgg_output)
            self.G_loss = -tf.reduce_mean(self.label * tf.log(self.predict + 0.00001))
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.label, 1)), "float"))
            self.G_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss, var_list=
                                                [var for var in tf.trainable_variables() if var.name.startswith('fc_')])
            self.saver = tf.train.Saver(max_to_keep=1000, var_list=[var for var in tf.trainable_variables()
                                                                    if var.name.startswith('fc_')])


    def extract_feature(self, data_train, label_train):
        feature_vgg0 = self.sess.run(self.vgg_output, feed_dict={self.input_image: data_train[0:1000]})
        feature_vgg1 = self.sess.run(self.vgg_output, feed_dict={self.input_image: data_train[1000:]})
        feature_vgg = np.concatenate((feature_vgg0, feature_vgg1), axis=0)
        del feature_vgg0, feature_vgg1
        feature_vgg_0 = np.zeros([501, feature_vgg.shape[1], feature_vgg.shape[2], feature_vgg.shape[3], types])
        for i in range(types):
            temp = np.mean(feature_vgg[label_train[:, i] == 1], axis=0)
            feature_vgg_0[:, :, :, :, i] = np.tile(temp, [501, 1, 1, 1])
        return feature_vgg_0


    def test(self):
        print("[*]Loading Model...")
        self.saver.restore(self.sess, "./checkpoint/ZSL/model10.ckpt")
        print("[*]Load successfully!")

        data_test = load_data(FLAGS)

        data_test.shape = -1, self.img_size, self.img_size, 1
        label_pred = self.sess.run(self.predict, feed_dict={self.input_image: data_test})
        sio.savemat('./result/pred.mat', {'label_pred': label_pred})

    def train(self):
        Epoches = self.Epoches
        Batch_Size = self.Batch_Size
        # load data
        self.sess.run(tf.global_variables_initializer())

        data_train, label_train, data_test, label_test, train_size, test_size = load_data(FLAGS)
        data_train.shape = -1, self.img_size, self.img_size, 1
        data_test.shape = -1, self.img_size, self.img_size, 1
        size_of_train = len(data_train)
        batch_idxs = size_of_train // Batch_Size

        acc_batch = np.zeros(batch_idxs * Epoches, dtype=float)
        acc_T72_epoch = np.zeros([Epoches])
        acc_test_epoch = np.zeros([Epoches])
        pred_syn = np.zeros([Epoches, 9, 501, types])

        feature_vgg = self.extract_feature(data_train, label_train)
        xx = np.arange(0, 1.002, 0.002)

        cnt = 0
        for epoch in range(Epoches):
            idx = np.random.permutation(size_of_train)
            g_loss = np.zeros([batch_idxs])
            for ind in range(batch_idxs):
                batch_data = data_train[idx[ind*Batch_Size:(ind+1)*Batch_Size]]
                batch_label = label_train[idx[ind*Batch_Size:(ind+1)*Batch_Size]]
                _ = self.sess.run(self.G_opt, feed_dict={self.label: batch_label, self.input_image: batch_data})
                acc_batch[cnt] = self.sess.run(self.acc, feed_dict={self.label: batch_label, self.input_image: batch_data})
                cnt += 1
                print("Epoch:%d || Batch:%d || Acc:%.4f || Cost Time:%.2f" % (epoch, ind, acc_batch[cnt-1],
                                                                              time.time() - st))

            self.saver.save(self.sess, FLAGS.dir + '/' + FLAGS.mode + "/model" + str(epoch) + ".ckpt")
            acc_test_epoch[epoch] = self.sess.run(self.acc, feed_dict={self.input_image: data_test[0:test_size, :, :, :],
                                                             self.label: label_test[0:test_size]})
            acc_T72_epoch[epoch] = self.sess.run(self.acc, feed_dict={self.input_image: data_test[test_size:],
                                                                      self.label: label_test[test_size:]})
            print("accuracy of Test: %.6f" % acc_test_epoch[epoch])
            print("accuracy of T72: %.6f" % acc_T72_epoch[epoch])

            a = 1.0 - np.tile(xx[:, np.newaxis, np.newaxis, np.newaxis], [1, 14, 14, 512])
            b = np.tile(xx[:, np.newaxis, np.newaxis, np.newaxis], [1, 14, 14, 512])
            indx_t72 = 0
            for i in range(types):
                if i != 7:
                    feature_vgg_syn = feature_vgg[:, :, :, :, 7] * a + feature_vgg[:, :, :, :, i] * b
                    pred_syn[epoch, indx_t72, :, :] = self.sess.run(self.predict, feed_dict={
                                                                                    self.vgg_output: feature_vgg_syn})
                    indx_t72 += 1

        scipy.io.savemat(FLAGS.result_dir + '/' + FLAGS.mode + '/g_loss.mat', {'loss_batch': loss_batch,
                                                                               'loss_epoch': loss_epoch,
                                                                               'loss_test_epoch': loss_test_epoch,
                                                                               'acc_batch': acc_batch,
                                                                               'acc_T72_epoch': acc_T72_epoch,
                                                                               'acc_test_epoch': acc_test_epoch,
                                                                               'pred_syn': pred_syn})

        self.saver.restore(sess, "./checkpoint/ZSL/model10.ckpt")
        test_pred = np.zeros([len(data_test), types])
        for ind in range(len(data_test) // Batch_Size):
            batch_data = data_test[ind * Batch_Size:(ind + 1) * Batch_Size]
            batch_data.shape = -1, img_size, img_size, 1
            test_pred[ind * Batch_Size:(ind + 1) * Batch_Size] = self.sess.run(self.predict, feed_dict={self.input_image: batch_data})
        batch_data = data_test[(ind + 1) * Batch_Size:]
        batch_data.shape = -1, img_size, img_size, 1
        test_pred[(ind + 1) * Batch_Size:] = self.sess.run(self.predict, feed_dict={self.input_image: batch_data})
        scipy.io.savemat(FLAGS.result_dir + '/' + FLAGS.mode + '/pred.mat', {'label_test': label_test,
                                                                             'label_pred': test_pred,
                                                                             'test_size': test_size})


def main(_):
    with tf.Session() as sess:
        zsl = zsl_model(sess)
        if FLAGS.is_test == True:
            zsl.test()
        else:
            zsl.train()


if __name__ == '__main__':
    tf.app.run()
