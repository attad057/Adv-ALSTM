import argparse
import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time
import pandas as pd
import dataframe_image as dfi

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops


    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)

from load import load_cla_data
from evaluator import evaluate, label

class AWLSTM:
    def __init__(self, data_path, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=256, gpu=False, tra_date='2014-01-02',
                 val_date='2015-08-03', tes_date='2015-10-01', att=0, hinge=0,
                 fix_init=0, adv=0, reload=0, dropout_activation_function = None, load_mappings=False):
        self.data_path = data_path
        self.model_path = model_path
        self.model_save_path = model_save_path
        # model parameters
        self.paras = copy.copy(parameters)

        if 'feat_dim' in self.paras:
            self.initial_feat_dim = self.paras['feat_dim']
        else:
            self.initial_feat_dim = 11
            self.paras['feat_dim'] = 11

        self.initial_seq = self.paras['seq']
        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        if att == 1:
            self.att = True
        else:
            self.att = False
        if hinge == 1:
            self.hinge = True
        else:
            self.hinge = False
        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False
        if adv == 1:
            self.adv_train = True
        else:
            self.adv_train = False
        if reload == 1:
            self.reload = True
        else:
            self.reload = False       

        self.dropout_activation_function = dropout_activation_function

        if dropout_activation_function != None:
            self.use_dropout_wrapper = True
        else:
            self.use_dropout_wrapper = False

        if self.use_dropout_wrapper == True:
            self.state_keep_prob = self.paras['state_keep_prob']
            self.input_keep_prob = 1.0
            self.output_keep_prob = 1.0

        self.tra_date = tra_date
        self.val_date = val_date
        self.tes_date = tes_date

        if load_mappings == False:
            # load data
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
                self.data_path,
                tra_date, val_date, tes_date, seq=self.paras['seq'], fea_dim=self.paras['feat_dim'], return_mappings=load_mappings
            )
            self.fea_dim = self.tra_pv.shape[2]
        else:
            # load data
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt, \
            self.val_mappings, self.tes_mappings = load_cla_data(
                self.data_path,
                tra_date, val_date, tes_date, seq=self.paras['seq'], fea_dim=self.paras['feat_dim'], return_mappings=load_mappings
            )
            self.fea_dim = self.tra_pv.shape[2]

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], \
               self.tra_wd[sta_ind:end_ind, :, :], \
               self.tra_gt[sta_ind:end_ind, :]

    def adv_part(self, adv_inputs):
        print('adversial part')
        if self.att:
            with tf.variable_scope('pre_fc'):
                self.fc_W = tf.get_variable(
                    'weights', dtype=tf.float32,
                    shape=[self.paras['unit'] * 2, 1],
                    initializer=tf.glorot_uniform_initializer()
                )
                self.fc_b = tf.get_variable(
                    'biases', dtype=tf.float32,
                    shape=[1, ],
                    initializer=tf.zeros_initializer()
                )
                if self.hinge:
                    pred = tf.nn.bias_add(
                        tf.matmul(adv_inputs, self.fc_W), self.fc_b
                    )
                else:
                    pred = tf.nn.sigmoid(
                        tf.nn.bias_add(tf.matmul(self.fea_con, self.fc_W),
                                       self.fc_b)
                    )
        else:
            # One hidden layer
            if self.hinge:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=None,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=tf.nn.sigmoid,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
        return pred

    def construct_graph(self):
        print('is pred_lstm')
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()
            if self.fix_init:
                tf.set_random_seed(123456)

            self.gt_var = tf.placeholder(tf.float32, [None, 1])
            self.pv_var = tf.placeholder(
                tf.float32, [None, self.paras['seq'], self.fea_dim]
            )
            self.wd_var = tf.placeholder(
                tf.float32, [None, self.paras['seq'], 5]
            )

            if self.use_dropout_wrapper == True:
                self.default_state_keep_prob = 1.0
                self.default_input_keep_prob = 1.0
                self.default_output_keep_prob = 1.0

                self.state_keep_prob_var = tf.placeholder_with_default(input=self.default_state_keep_prob, shape=(), name='state_keep_prob')
                self.input_keep_prob_var = tf.placeholder_with_default(input=self.default_input_keep_prob, shape=(), name='input_keep_prob')
                self.output_keep_prob_var = tf.placeholder_with_default(input=self.default_output_keep_prob, shape=(), name='output_keep_prob')

                self.lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.paras['unit']), 
                state_keep_prob=self.state_keep_prob_var, 
                output_keep_prob=self.output_keep_prob_var,
                input_keep_prob=self.input_keep_prob_var) 
            else: 
                self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.paras['unit'])

            self.in_lat = tf.layers.dense(
                self.pv_var, units=self.fea_dim,
                activation=tf.nn.tanh, name='in_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            self.outputs, _ = tf.nn.dynamic_rnn(
                # self.outputs, _ = tf.nn.static_rnn(
                self.lstm_cell, self.in_lat, dtype=tf.float32
                # , initial_state=ini_sta
            )

            self.loss = 0
            self.adv_loss = 0
            self.l2_norm = 0
            if self.att:
                with tf.variable_scope('lstm_att') as scope:
                    self.av_W = tf.get_variable(
                        name='att_W', dtype=tf.float32,
                        shape=[self.paras['unit'], self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )
                    self.av_b = tf.get_variable(
                        name='att_h', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.zeros_initializer()
                    )
                    self.av_u = tf.get_variable(
                        name='att_u', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )

                    self.a_laten = tf.tanh(
                        tf.tensordot(self.outputs, self.av_W,
                                     axes=1) + self.av_b)
                    self.a_scores = tf.tensordot(self.a_laten, self.av_u,
                                                 axes=1,
                                                 name='scores')
                    self.a_alphas = tf.nn.softmax(self.a_scores, name='alphas')

                    self.a_con = tf.reduce_sum(
                        self.outputs * tf.expand_dims(self.a_alphas, -1), 1)
                    self.fea_con = tf.concat(
                        [self.outputs[:, -1, :], self.a_con],
                        axis=1)
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.fea_con)
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(self.loss, [self.fea_con])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv, axis=1)
                        self.adv_pv_var = self.fea_con + \
                                          self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var, self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var, self.adv_pred)
            else:
                with tf.variable_scope('lstm_att') as scope:
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.outputs[:, -1, :])
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(self.loss, [self.outputs[:, -1, :]])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv,
                                                            axis=1)
                        self.adv_pv_var = self.outputs[:, -1, :] + \
                                          self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var,
                                                                 self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var,
                                                               self.adv_pred)

            # regularizer
            self.tra_vars = tf.trainable_variables('lstm_att/pre_fc')
            for var in self.tra_vars:
                self.l2_norm += tf.nn.l2_loss(var)

            self.obj_func = self.loss + \
                            self.paras['alp'] * self.l2_norm + \
                            self.paras['bet'] * self.adv_loss

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.paras['lr']
            ).minimize(self.obj_func)

    def get_latent_rep(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1

        tr_lat_rep = np.zeros([bat_count * self.batch_size, self.paras['unit'] * 2],
                              dtype=np.float32)
        tr_gt = np.zeros([bat_count * self.batch_size, 1], dtype=np.float32)
        for j in range(bat_count):
            pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
            feed_dict = {
                self.pv_var: pv_b,
                self.wd_var: wd_b,
                self.gt_var: gt_b
            }
            lat_rep, cur_obj, cur_loss, cur_l2, cur_al = sess.run(
                (self.fea_con, self.obj_func, self.loss, self.l2_norm,
                 self.adv_loss),
                feed_dict
            )
            print(lat_rep.shape)
            tr_lat_rep[j * self.batch_size: (j + 1) * self.batch_size, :] = lat_rep
            tr_gt[j * self.batch_size: (j + 1) * self.batch_size,:] = gt_b

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_lat_rep, val_pre = sess.run(
            (self.loss, self.fea_con, self.pred), feed_dict
        )
        cur_val_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_val_perf)

        sess.close()
        tf.reset_default_graph()
        np.savetxt(self.model_save_path + '_val_lat_rep.csv', val_lat_rep)
        np.savetxt(self.model_save_path + '_tr_lat_rep.csv', tr_lat_rep)
        np.savetxt(self.model_save_path + '_val_gt.csv', self.val_gt)
        np.savetxt(self.model_save_path + '_tr_gt.csv', tr_gt)

    def monte_carlo_softmax(self, pre_np_arr):
        #2555, 3000
        pre_arr_s = np.reshape(np.transpose(pre_np_arr), (pre_np_arr.shape[1], pre_np_arr.shape[0]))
        pre_exp = np.exp(pre_arr_s)
        pre_exp_sum = np.reshape(np.sum(pre_exp, axis = 1), (pre_np_arr.shape[1], pre_np_arr.shape[2]))
        soft_max = pre_exp/pre_exp_sum

        # #2555
        # label_1_count = np.count_nonzero(pre_arr_s, axis = 1).astype(np.float32)
        # label_0_count = (pre_arr_s.shape[1] - label_1_count).astype(np.float32)

        # #2555, 3000
        # for idx, x in enumerate(pre_arr_s):
        #     has_1 = False
        #     has_0 = False
        #     for idx2, x2 in enumerate(x):
        #         if x2 == 1:
        #             soft_max[idx][idx2] = np.round(np.multiply(soft_max[idx][idx2], label_1_count[idx]), 4)
        #             has_1 = True
        #             if soft_max[idx][idx2] == 1:
        #                 has_0 = True
        #         else:
        #             soft_max[idx][idx2] = np.round(np.multiply(soft_max[idx][idx2], label_0_count[idx]), 4)
        #             has_0 = True
        #             if soft_max[idx][idx2] == 1:
        #                 has_1 = True
        #         if has_1 == True and has_0 == True:
        #             break
                    
        ind_columns = np.argmax(soft_max, axis=1)

        #take best indexes from pre_np_arr
        ind_rows = np.arange(pre_np_arr.shape[1])
        pre = np.reshape(pre_arr_s[ind_rows, ind_columns], (pre_np_arr.shape[1], pre_np_arr.shape[2]))
        return pre, np.max(soft_max, axis=1)

    def monte_carlo_average(self, pre_np_arr):
        #2555, 3000
        pre_arr_s = np.reshape(np.transpose(pre_np_arr), (pre_np_arr.shape[1], pre_np_arr.shape[0]))
        pre_avg = np.mean(pre_arr_s, axis = 1)
        pre_avg_std = np.std(pre_arr_s, axis = 1)
        pre_avg_round = np.reshape(np.round(pre_avg), (pre_np_arr.shape[1], pre_np_arr.shape[2]))

        #take best indexes from pre_np_arr
        # ind_rows = np.arange(pre_np_arr.shape[1])
        # pre = np.reshape(pre_arr_s[ind_rows, ind_columns], (pre_np_arr.shape[1], pre_np_arr.shape[2]))
        return pre_avg_round, pre_avg_std

    def predict_adv(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        tra_perf = None
        adv_perf = None
        for j in range(bat_count):
            pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
            feed_dict = {
                self.pv_var: pv_b,
                self.wd_var: wd_b,
                self.gt_var: gt_b
            }
            cur_pre, cur_adv_pre, cur_obj, cur_loss, cur_l2, cur_al = sess.run(
                (self.pred, self.adv_pred, self.obj_func, self.loss, self.l2_norm,
                 self.adv_loss),
                feed_dict
            )
            cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
            cur_adv_perf = evaluate(cur_adv_pre, gt_b, self.hinge)
            if tra_perf is None:
                tra_perf = copy.copy(cur_tra_perf)
            else:
                for metric in tra_perf.keys():
                    tra_perf[metric] = tra_perf[metric] + cur_tra_perf[metric]
            if adv_perf is None:
                adv_perf = copy.copy(cur_adv_perf)
            else:
                for metric in adv_perf.keys():
                    adv_perf[metric] = adv_perf[metric] + cur_adv_perf[metric]
        for metric in tra_perf.keys():
            tra_perf[metric] = tra_perf[metric] / bat_count
            adv_perf[metric] = adv_perf[metric] / bat_count

        print('Clean samples performance:', tra_perf)
        print('Adversarial samples performance:', adv_perf)

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre, val_adv_pre = sess.run(
            (self.loss, self.pred, self.adv_pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per clean:', cur_valid_perf)
        adv_valid_perf = evaluate(val_adv_pre, self.val_gt, self.hinge)
        print('\tVal per adversarial:', adv_valid_perf)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre, tes_adv_pre = sess.run(
            (self.loss, self.pred, self.adv_pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per clean:', cur_test_perf)
        adv_test_perf = evaluate(tes_adv_pre, self.tes_gt, self.hinge)
        print('\tTest per adversarial:', adv_test_perf)

        sess.close()
        tf.reset_default_graph()

    def predict_record(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)
        np.savetxt(self.model_save_path + '_val_prediction.csv', val_pre)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        np.savetxt(self.model_save_path + '_tes_prediction.csv', tes_pre)
        sess.close()
        tf.reset_default_graph()

    def test(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        sess.close()
        tf.reset_default_graph()

    def train(self, tune_para=False, return_perf=False, return_pred=True, save_model = True):
        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        best_test_perf = {
            'acc': 0, 'mcc': -2
        }

        self.construct_graph()
        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1

        epochs_arr = [*range(self.epochs + 1)][1:]

        for i in epochs_arr:

            t1 = time()
            # first_batch = True
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_adv = 0.0

            for j in range(bat_count):
                pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.wd_var: wd_b,
                    self.gt_var: gt_b
                }

                cur_pre, cur_obj, cur_loss, cur_l2, cur_al, batch_out = sess.run(
                    (self.pred, self.obj_func, self.loss, self.l2_norm, self.adv_loss,
                    self.optimizer),
                    feed_dict
                )

                tra_loss += cur_loss
                tra_obj += cur_obj
                l2 += cur_l2
                tra_adv += cur_al
            print('----->>>>> Training:', tra_obj / bat_count,
                tra_loss / bat_count, l2 / bat_count, tra_adv / bat_count)

            if not tune_para:
                tra_loss = 0.0
                tra_obj = 0.0
                l2 = 0.0
                tra_acc = 0.0
                for j in range(bat_count):
                    pv_b, wd_b, gt_b = self.get_batch(
                        j * self.batch_size)
                    feed_dict = {
                        self.pv_var: pv_b,
                        self.wd_var: wd_b,
                        self.gt_var: gt_b
                    }
                    cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
                        (self.obj_func, self.loss, self.l2_norm, self.pred),
                        feed_dict
                    )
                    cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
                    tra_loss += cur_loss
                    l2 += cur_l2
                    tra_obj += cur_obj
                    tra_acc += cur_tra_perf['acc']
                print('Training:', tra_obj / bat_count, tra_loss / bat_count,
                    l2 / bat_count, '\tTrain per:', tra_acc / bat_count)

            # test on validation set
            feed_dict = {
                self.pv_var: self.val_pv,
                self.wd_var: self.val_wd,
                self.gt_var: self.val_gt
            }
               
            val_loss, val_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )

            cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge, additional_metrics=True)
            cur_valid_perf_p = {
                'acc': cur_valid_perf['acc'],
                'mcc': cur_valid_perf['mcc']
            }
            print('\tVal per:', cur_valid_perf_p, '\tVal loss:', val_loss)

            # test on testing set
            feed_dict = {
                self.pv_var: self.tes_pv,
                self.wd_var: self.tes_wd,
                self.gt_var: self.tes_gt
            }

            test_loss, tes_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )
       
            cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge, additional_metrics=True)

            cur_test_perf_p = {
                'acc': cur_test_perf['acc'],
                'mcc': cur_test_perf['mcc']
            }        
  
            print('\tTest per:', cur_test_perf_p, '\tTest loss:', test_loss)   

            if cur_valid_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_pred = copy.copy(val_pre)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_pred = copy.copy(tes_pre)
                if not tune_para and save_model == True:
                    saver.save(sess, self.model_save_path)
            self.tra_pv, self.tra_wd, self.tra_gt = shuffle(
                self.tra_pv, self.tra_wd, self.tra_gt, random_state=0
            )
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        
        # training performance

        best_valid_perf_p = {
            'acc': best_valid_perf['acc'],
            'mcc': best_valid_perf['mcc']
        }

        best_test_perf_p = {
            'acc': best_test_perf['acc'],
            'mcc': best_test_perf['mcc']
        }
        print('\nBest Valid performance:', best_valid_perf_p)
        print('\tBest Test performance:', best_test_perf_p)
        sess.close()
        tf.reset_default_graph()

        if return_pred == True and return_perf == True:
            return best_valid_perf, best_test_perf, best_valid_pred, best_test_pred
        elif tune_para or return_perf == True:
            return best_valid_perf, best_test_perf
        else:
            return best_valid_pred, best_test_pred

    def train_monte_carlo_dropout(self, tune_para=False, return_perf=False, return_pred=True, iterations=1, evaluate_last_epoch_only=False):
        iterations_arr = [*range(iterations + 1)][1:]
        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        best_test_perf = {
            'acc': 0, 'mcc': -2
        }

        self.construct_graph()
        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1

        epochs_arr = [*range(self.epochs + 1)][1:]

        for i in epochs_arr:

            t1 = time()
            # first_batch = True
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_adv = 0.0

            for j in range(bat_count):
                pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.wd_var: wd_b,
                    self.gt_var: gt_b,
                    self.state_keep_prob_var: self.default_state_keep_prob,
                    self.input_keep_prob_var: self.default_input_keep_prob,
                    self.output_keep_prob_var: self.default_output_keep_prob
                }

                cur_pre, cur_obj, cur_loss, cur_l2, cur_al, batch_out = sess.run(
                    (self.pred, self.obj_func, self.loss, self.l2_norm, self.adv_loss,
                    self.optimizer),
                    feed_dict
                )

                tra_loss += cur_loss
                tra_obj += cur_obj
                l2 += cur_l2
                tra_adv += cur_al
            print('----->>>>> Training:', tra_obj / bat_count,
                tra_loss / bat_count, l2 / bat_count, tra_adv / bat_count, '\t state_keep_prob', self.state_keep_prob)

            if not tune_para:
                tra_loss = 0.0
                tra_obj = 0.0
                l2 = 0.0
                tra_acc = 0.0
                for j in range(bat_count):
                    pv_b, wd_b, gt_b = self.get_batch(
                        j * self.batch_size)
                    feed_dict = {
                        self.pv_var: pv_b,
                        self.wd_var: wd_b,
                        self.gt_var: gt_b,
                        self.state_keep_prob_var: self.default_state_keep_prob,
                        self.input_keep_prob_var: self.default_input_keep_prob,
                        self.output_keep_prob_var: self.default_output_keep_prob
                    }
                    cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
                        (self.obj_func, self.loss, self.l2_norm, self.pred),
                        feed_dict
                    )
                    cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
                    tra_loss += cur_loss
                    l2 += cur_l2
                    tra_obj += cur_obj
                    tra_acc += cur_tra_perf['acc']
                print('Training:', tra_obj / bat_count, tra_loss / bat_count,
                    l2 / bat_count, '\tTrain per:', tra_acc / bat_count)

            # test on validation set
            val_l_arr = []
            feed_dict = {
                self.pv_var: self.val_pv,
                self.wd_var: self.val_wd,
                self.gt_var: self.val_gt,
                self.state_keep_prob_var: self.state_keep_prob,
                self.input_keep_prob_var: self.input_keep_prob,
                self.output_keep_prob_var: self.output_keep_prob
            }
            
            #We evaluate at last epoch due to performance issues
            if i == self.epochs or evaluate_last_epoch_only == False:
                for r in iterations_arr:
                    val_loss, val_pre = sess.run(
                        (self.loss, self.pred), feed_dict
                    )               
        
                val_l_arr.append(label(self.hinge, val_pre))

                #shape (3720, 10)
                val_l_np_arr = np.array(val_l_arr)

                if self.dropout_activation_function == 'avg':
                    val_pre, val_pre_prob = self.monte_carlo_average(val_l_np_arr)
                elif self.dropout_activation_function == 'softmax':
                    val_pre, val_pre_prob = self.monte_carlo_softmax(val_l_np_arr)

                cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge, additional_metrics=True)
                cur_valid_perf['prob_arr'] = val_pre_prob
                cur_valid_perf_p = {
                    'acc': cur_valid_perf['acc'],
                    'mcc': cur_valid_perf['mcc']
                }
                print('\tVal per:', cur_valid_perf_p, '\tVal loss:', val_loss, '\tVal state_keep_prob:', self.state_keep_prob)

                # test on testing set
                feed_dict = {
                    self.pv_var: self.tes_pv,
                    self.wd_var: self.tes_wd,
                    self.gt_var: self.tes_gt,
                    self.state_keep_prob_var: self.state_keep_prob,
                    self.input_keep_prob_var: self.input_keep_prob,
                    self.output_keep_prob_var: self.output_keep_prob
                }

                test_l_arr = []
                for r in iterations_arr:
                    test_loss, tes_pre = sess.run(
                        (self.loss, self.pred), feed_dict
                    )
                    test_l_arr.append(label(self.hinge, tes_pre))

                #shape (3720, 10)
                test_l_np_arr = np.array(test_l_arr)

                if self.dropout_activation_function == 'avg':
                    tes_pre, test_pre_prob = self.monte_carlo_average(test_l_np_arr)
                elif self.dropout_activation_function == 'softmax':
                    tes_pre, test_pre_prob = self.monte_carlo_softmax(test_l_np_arr)

                cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge, additional_metrics=True)
                cur_test_perf['prob_arr'] = test_pre_prob

                cur_test_perf_p = {
                    'acc': cur_test_perf['acc'],
                    'mcc': cur_test_perf['mcc']
                }        
    
                print('\tTest per:', cur_test_perf_p, '\tTest loss:', test_loss, '\tTest state_keep_prob:', self.state_keep_prob)
            
                if cur_valid_perf['acc'] > best_valid_perf['acc']:
                    best_valid_perf = copy.copy(cur_valid_perf)
                    best_valid_pred = copy.copy(val_pre)
                    best_test_perf = copy.copy(cur_test_perf)
                    best_test_pred = copy.copy(tes_pre)
                    if not tune_para:
                        saver.save(sess, self.model_save_path)
                self.tra_pv, self.tra_wd, self.tra_gt = shuffle(
                    self.tra_pv, self.tra_wd, self.tra_gt, random_state=0
                )
                t4 = time()
                print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
            
                # training performance

                best_valid_perf_p = {
                    'acc': best_valid_perf['acc'],
                    'mcc': best_valid_perf['mcc']
                }

                best_test_perf_p = {
                    'acc': best_test_perf['acc'],
                    'mcc': best_test_perf['mcc']
                }
                print('\nBest Valid performance:', best_valid_perf_p)
                print('\tBest Test performance:', best_test_perf_p)
        sess.close()
        tf.reset_default_graph()

        if return_pred == True and return_perf == True:
            return best_valid_perf, best_test_perf, best_valid_pred, best_test_pred
        elif tune_para or return_perf == True:
            return best_valid_perf, best_test_perf
        else:
            return best_valid_pred, best_test_pred
            
    def train_convergence_monte_carlo_dropout(self, tune_para=False, benchmark_increment = 250, max_iteration = 500000):
        # iterations_benchmark_arr.sort()
        # max_iteration = iterations_benchmark_arr[-1]
        iterations_arr = [*range(max_iteration + 1)][1:]
        #val_std_arr = []
        #tes_std_arr = []

        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        best_test_perf = {
            'acc': 0, 'mcc': -2
        }

        self.construct_graph()
        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1

        #epochs_arr = [*range(2)][1:]
        epochs_arr = [*range(self.epochs + 1)][1:]
        val_rt_curr = np.zeros(self.val_gt.shape[0])
        tes_rt_curr = np.zeros(self.tes_gt.shape[0])
        val_std_p_arr = []
        tes_std_p_arr = []

        for i in epochs_arr:
    
            t1 = time()
            # first_batch = True
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_adv = 0.0

            for j in range(bat_count):
                pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.wd_var: wd_b,
                    self.gt_var: gt_b,
                    self.state_keep_prob_var: self.default_state_keep_prob,
                    self.input_keep_prob_var: self.default_input_keep_prob,
                    self.output_keep_prob_var: self.default_output_keep_prob
                }

                cur_pre, cur_obj, cur_loss, cur_l2, cur_al, batch_out = sess.run(
                    (self.pred, self.obj_func, self.loss, self.l2_norm, self.adv_loss,
                    self.optimizer),
                    feed_dict
                )

                tra_loss += cur_loss
                tra_obj += cur_obj
                l2 += cur_l2
                tra_adv += cur_al
            print('----->>>>> Training:', tra_obj / bat_count,
                tra_loss / bat_count, l2 / bat_count, tra_adv / bat_count, '\t state_keep_prob', self.state_keep_prob)

            if not tune_para:
                tra_loss = 0.0
                tra_obj = 0.0
                l2 = 0.0
                tra_acc = 0.0
                for j in range(bat_count):
                    pv_b, wd_b, gt_b = self.get_batch(
                        j * self.batch_size)
                    feed_dict = {
                        self.pv_var: pv_b,
                        self.wd_var: wd_b,
                        self.gt_var: gt_b,
                        self.state_keep_prob_var: self.default_state_keep_prob,
                        self.input_keep_prob_var: self.default_input_keep_prob,
                        self.output_keep_prob_var: self.default_output_keep_prob
                    }
                    cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
                        (self.obj_func, self.loss, self.l2_norm, self.pred),
                        feed_dict
                    )
                    cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
                    tra_loss += cur_loss
                    l2 += cur_l2
                    tra_obj += cur_obj
                    tra_acc += cur_tra_perf['acc']
                print('Training:', tra_obj / bat_count, tra_loss / bat_count,
                    l2 / bat_count, '\tTrain per:', tra_acc / bat_count)
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
            self.tra_pv, self.tra_wd, self.tra_gt = shuffle(
                self.tra_pv, self.tra_wd, self.tra_gt, random_state=0
            )
        # test on validation set
        val_l_arr = []
        val_pre_arr = []
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt,
            self.state_keep_prob_var: self.state_keep_prob,
            self.input_keep_prob_var: self.input_keep_prob,
            self.output_keep_prob_var: self.output_keep_prob
        }
        
        #iterations_arr = [*range(100000 + 1)][1:]
        val_l_np_arr = np.array([])
        for r in iterations_arr:
            val_loss, val_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )

            val_pre_arr.append(val_pre)

            print('val: ' + str(r))

            if (r >= benchmark_increment and r % benchmark_increment == 0):

                val_l_np_arr = np.array(val_pre_arr[0: r])
                pre_arr_s = np.reshape(np.transpose(val_l_np_arr), (val_l_np_arr.shape[1], val_l_np_arr.shape[0]))
                val_std = np.std(pre_arr_s, axis = 1)
                val_std_err = val_std / np.sqrt(np.array([pre_arr_s.shape[1]]))
                val_rt_diff = (val_std_err - val_rt_curr) 

                if r == benchmark_increment or np.any(val_rt_diff > 0):
                    rt = val_std_err + ((val_std_err / 100) * 5)
                    val_rt_curr = rt

                elif r == max_iteration:
                    val_std_p_arr.append({'avg std err': np.mean(val_std_err), 'epoch': i, 'iterations': str(r-benchmark_increment), 'ci': None})
                    print('reached maximum ' + str(r) + ' iterations, did not pass 95% confidence interval')
                    break;
                else:
                    val_std_p_arr.append({'avg std err': np.mean(val_std_err), 'epoch': i, 'iterations': str(r-benchmark_increment), 'ci': 95})
                    print('within 95% confidence interval ' + str(r) + ' iterations')
                    break;

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt,
            self.state_keep_prob_var: self.state_keep_prob,
            self.input_keep_prob_var: self.input_keep_prob,
            self.output_keep_prob_var: self.output_keep_prob
        }

        test_pre_arr = []

        for r in iterations_arr:
            test_loss, tes_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )

            test_pre_arr.append(tes_pre)
            print('tes: ' + str(r))
            if (r >= benchmark_increment and r % benchmark_increment == 0):
                tes_l_np_arr = np.array(test_pre_arr[0: r])
                pre_arr_s = np.reshape(np.transpose(tes_l_np_arr), (tes_l_np_arr.shape[1], tes_l_np_arr.shape[0]))
                tes_std = np.std(pre_arr_s, axis = 1)
                tes_std_err = tes_std / np.sqrt(np.array([pre_arr_s.shape[1]]))

                tes_rt_diff = (tes_std_err - tes_rt_curr) 

                if r == benchmark_increment or np.any(tes_rt_diff > 0):
                    rt = tes_std_err + ((tes_std_err / 100) * 5)
                    tes_rt_curr = rt

                elif r == max_iteration:
                    tes_std_p_arr.append({'avg std err': np.mean(tes_std_err), 'epoch': i, 'iterations': str(r), 'ci': None})
                    print('reached maximum ' + str(r) + ' iterations, did not pass 95% confidence interval')
                    break;
                else:
                    tes_std_p_arr.append({'avg std err': np.mean(tes_std_err), 'epoch': i, 'iterations': str(r-benchmark_increment), 'ci': 95})
                    print('within 95% confidence interval ' + str(r-benchmark_increment) + ' iterations')
                    break;
        
        sess.close()
        tf.reset_default_graph()

        return val_std_p_arr, tes_std_p_arr
   
    def train_ensemble(self, tune_para=False, return_perf=False, return_pred=True, param_iterations = 10):
        parameters = copy.copy(self.paras)
        start_feat_dim = self.initial_feat_dim
        start_seq = self.initial_seq
        parameters['seq'] = start_seq
        parameters['feat_dim'] = start_feat_dim 
        decay_seq = np.round(parameters['seq'] / param_iterations )
        #decay_feat_dim = np.round(parameters['feat_dim'] / param_iterations)

        ensemble_model_results = []

        for r in [*range(param_iterations + 1)][1:]:
            self.update_model(parameters)
            best_valid_perf, best_test_perf, best_valid_pred, best_test_pred = pure_LSTM.train(return_perf=True, return_pred=True, tune_para=tune_para, save_model=False)
            best_valid_pred = label(self.hinge, best_valid_pred)
            best_test_pred = label(self.hinge, best_test_pred)

            ensemble_model_results.append({
                'best_valid_perf': best_valid_perf, 
                'best_test_perf': best_test_perf,
                'best_valid_pred': best_valid_pred,                
                'best_test_pred': best_test_pred,
                'seq': parameters['seq'],
                'feat_dim': parameters['feat_dim']
                })
            #for r2 in [*range(param_iterations)][1:]:
                #if (parameters['feat_dim'] - decay_feat_dim >= 1):
                #   parameters['feat_dim'] = int(parameters['feat_dim']  - decay_feat_dim)
            #        self.update_model(parameters)
            #        best_valid_perf, best_test_perf, best_valid_pred, best_test_pred = pure_LSTM.train(return_perf=True, return_pred=True, tune_para=tune_para, save_model=False)
            #        best_valid_pred = label(self.hinge, best_valid_pred)
            #        best_test_pred = label(self.hinge, best_test_pred)

            #        ensemble_model_results.append({
            #        'best_valid_perf': best_valid_perf, 
            #        'best_test_perf': best_test_perf,
            #        'best_valid_pred': best_valid_pred,                
            #        'best_test_pred': best_test_pred,
            #        'seq': parameters['seq'],
            #        'feat_dim': parameters['feat_dim']
            #        })
            if (parameters['seq'] - decay_seq >= 1):
                parameters['seq'] = int(parameters['seq'] - decay_seq)

            #parameters['feat_dim'] = start_feat_dim 

        #100,2555, 1
        best_valid_pred_list = np.array(list(map(lambda x: x['best_valid_pred'], ensemble_model_results)))
        #best_valid_pred_list = np.random.randint(2, size = [25, 2555, 1])
        val_pre_arr_s = np.transpose(np.reshape(best_valid_pred_list, (best_valid_pred_list.shape[0], best_valid_pred_list.shape[1])))
        val_pre_arr = []
        val_std_arr = []
        for ind, i in enumerate(val_pre_arr_s):
            val_std_arr.append(np.std(val_pre_arr_s[ind]))
            pre_unique, pre_count = np.unique(i, return_counts=True)
            pre_argmax = np.argmax(pre_count)
            pre_max = pre_unique[pre_argmax]
            val_pre_arr.append(np.array([pre_max]))

        val_pre = np.array(val_pre_arr)
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge, additional_metrics=True)
        cur_valid_perf['prob_arr'] = val_std_arr

        best_test_pred_list = np.array(list(map(lambda x: x['best_test_pred'], ensemble_model_results)))
        #best_test_pred_list = np.random.randint(2, size = [25, 3720, 1])
        test_pre_arr_s = np.transpose(np.reshape(best_test_pred_list, (best_test_pred_list.shape[0], best_test_pred_list.shape[1])))

        test_pre_arr = []
        test_std_arr = []
        for ind, i in enumerate(test_pre_arr_s):
            test_std_arr.append(np.std(test_pre_arr_s[ind]))
            pre_unique, pre_count = np.unique(i, return_counts=True)
            pre_argmax = np.argmax(pre_count)
            pre_max = pre_unique[pre_argmax]
            test_pre_arr.append(np.array([pre_max]))

        test_pre = np.array(test_pre_arr)
        cur_test_perf = evaluate(test_pre, self.tes_gt, self.hinge, additional_metrics=True)
        cur_test_perf['prob_arr'] = test_std_arr

        ensemble_model_results = np.array(ensemble_model_results)
        if return_perf == True and return_pred == True:
            return ensemble_model_results, cur_valid_perf, cur_test_perf, val_pre, test_pre
        elif return_perf == True:
            return ensemble_model_results, cur_valid_perf, cur_test_perf
        elif return_pred == True:
            return ensemble_model_results, val_pre, test_pre
        else:
            return ensemble_model_results

    def update_model(self, parameters):
        data_update = False
        if not parameters['seq'] == self.paras['seq'] or not parameters['feat_dim'] == self.paras['feat_dim']:
            data_update = True

        for name, value in parameters.items():
            self.paras[name] = value
        if data_update:
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
                self.data_path,
                self.tra_date, self.val_date, self.tes_date, seq=self.paras['seq'], fea_dim=self.paras['feat_dim']
            )
            self.fea_dim = self.tra_pv.shape[2]
        return True

if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./data/stocknet-dataset/price/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                        help='epsilon to control the scale of noise')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-q', '--model_path', help='path to load model',
                        type=str, default='./saved_model/acl18_alstm/exp')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                        default='./tmp/model')
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, pred, replicate')
    parser.add_argument('-m', '--model', type=str, default='pure_lstm',
                        help='pure_lstm, di_lstm, att_lstm, week_lstm, aw_lstm')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-a', '--att', type=int, default=1,
                        help='use attention model')
    parser.add_argument('-w', '--week', type=int, default=0,
                        help='use week day data')
    parser.add_argument('-v', '--adv', type=int, default=0,
                        help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                        help='use hinge lose')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    parser.add_argument('-sp', '--state_keep_prob', type=float, default=1,
                    help='Apply dropout to each layer')
    args = parser.parse_args()

    args.action = 'experiment1_dropout_convergence'
    #args.dropout_wrapper = 1
    dataset = 'stocknet' if 'stocknet' in args.path else 'kdd17' if 'kdd17' in args.path else ''
    method = ''

    if args.action == 'replication':
        predefined_args = [ 
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 0,
            #-l
            'seq': 10,
            #-u
            'unit': 32,
            #-l2
            'alpha_l2': 10,
            #-f
            'fix_init': 1,
            #-v
            'adv': 0,
            #-rl
            'reload': 0,
            #-la
            'beta_adv': 1e-2,
            #-le
            'epsilon_adv': 1e-2,
            #-q
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'LSTM',
            'dataset': 'stocknet'
        }, 
        {
        #-p
        'path': './data/stocknet-dataset/price/ourpped',
        #-a
        'att': 1,
        #-l
        'seq': 5,
        #-u
        'unit': 4,
        #-l2
        'alpha_l2': 1,
        #-f
        'fix_init': 1,
        #-v
        'adv': 0,
        #-rl
        'reload': 0,
        #-la
        'beta_adv': 1e-2,
        #-le
        'epsilon_adv': 1e-2,
        #-q
        'model_path': './saved_model/acl18_alstm/exp',
        'model_save_path': './tmp/model',
        'method': 'ALSTM',
        'dataset': 'stocknet'
    },
    {
        #-p
        'path': './data/stocknet-dataset/price/ourpped',
        #-a
        'att': 1,
        #-l
        'seq': 5,
        #-u
        'unit': 4,
        #-l2
        'alpha_l2': 1,
        #-f
        'fix_init': 0,
        #-v
        'adv': 1,
        #-rl
        'reload': 1,
        #-la
        'beta_adv': 0.01,
        #-le
        'epsilon_adv': 0.05,
        'model_path': './saved_model/acl18_alstm/exp',
        'model_save_path': './tmp/model',
        'method': 'Adv-ALSTM',
        'dataset': 'stocknet'
    },   
    {
        #-p
        'path': './data/kdd17/ourpped/',
        #-a
        'att': 0,
        #-l
        'seq': 5,
        #-u
        'unit': 4,
        #-l2
        'alpha_l2': 0.001,
        #-f
        'fix_init': 1,
        #-v
        'adv': 0,
        #-rl
        'reload': 0,
        #-la
        'beta_adv': 1e-2,
        #-le
        'epsilon_adv': 1e-2,
        #-q
        'model_path': './saved_model/kdd17_alstm/model',
        'model_save_path': './tmp/model',
        'method': 'LSTM',
        'dataset': 'kdd17'
    },  
    {
        #-p
        'path': './data/kdd17/ourpped/',
        #-a
        'att': 1,
        #-l
        'seq': 15,
        #-u
        'unit': 16,
        #-l2
        'alpha_l2': 0.001,
        #-f
        'fix_init': 1,
        #-v
        'adv': 0,
        #-rl
        'reload': 0,
        #-la
        'beta_adv': 1e-2,
        #-le
        'epsilon_adv': 1e-2,
        #-q
        'model_path': './saved_model/kdd17_alstm/model',
        'model_save_path': './tmp/model',
        'method': 'ALSTM',
        'dataset': 'kdd17'
    },
    {
        #-p
        'path': './data/kdd17/ourpped/',
        #-a
        'att': 1,
        #-l
        'seq': 15,
        #-u
        'unit': 16,
        #-l2
        'alpha_l2': 0.001,
        #-f
        'fix_init': 1,
        #-v
        'adv': 1,
        #-rl
        'reload': 1,
        #-la
        'beta_adv': 0.05,
        #-le
        'epsilon_adv':  0.001,
        'model_path': './saved_model/kdd17_alstm/model',
        'model_save_path': './tmp/model',
        'method': 'Adv-ALSTM',
        'dataset': 'kdd17'
    }]
        
        perf_df = None
        perf_df2 = None
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            args.model_save_path = pre['model_save_path']
            method = pre['method']
            dataset = pre['dataset']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected dataset: %s' % dataset)
                exit(0)
            print(args)
            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate)
            }
                    
            pure_LSTM = AWLSTM(
                data_path=args.path,
                model_path=args.model_path,
                model_save_path=args.model_save_path,
                parameters=parameters,
                steps=args.step,
                epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                reload=args.reload,
                dropout_activation_function = None
            )

            runs = 5
            runs_arr = [*range(runs + 1)][1:]    
            pred_valid_arr = []
            pred_test_arr = []

            for r in runs_arr:
                best_valid_perf, best_test_perf = pure_LSTM.train(return_perf=True, return_pred=False)
                pred_valid_arr.append(best_valid_perf)
                pred_test_arr.append(best_test_perf)
                perf_dict = {
                        'method': [method],
                        'dataset': [dataset],
                        'test acc': [best_test_perf['acc'] * 100],
                        'test mcc': [best_test_perf['mcc']],
                        'valid acc': [best_valid_perf['acc'] * 100],
                        'valid mcc': [best_valid_perf['mcc']],
                        'run': [r]
                    }

                df = pd.DataFrame(perf_dict)
                if perf_df is None:
                    perf_df = df
                else:
                    perf_df = pd.concat([perf_df, df])            

                if not os.path.exists('replication'):
                    os.mkdir('replication')
                perf_df.to_csv('replication/perf_results.csv', index = False)
                dfi.export(perf_df,"replication/perf_results.png")

            valid_acc_list = list(map(lambda x: x['acc'], pred_valid_arr))
            valid_mcc_list = list(map(lambda x: x['mcc'], pred_valid_arr))
            valid_ll_list = list(map(lambda x: x['ll'], pred_valid_arr))

            avg_valid_acc = np.average(np.array(valid_acc_list)) * 100
            avg_valid_mcc = np.average(np.array(valid_mcc_list))
            avg_valid_ll = np.average(np.array(valid_ll_list))
            std_valid_ll = np.std(np.array(valid_ll_list), ddof=1) / np.sqrt(np.size(np.array(valid_ll_list)))

            test_acc_list = list(map(lambda x: x['acc'], pred_test_arr))
            test_mcc_list = list(map(lambda x: x['mcc'], pred_test_arr))
            test_ll_list = list(map(lambda x: x['ll'], pred_test_arr))

            avg_test_acc = np.average(np.array(test_acc_list)) * 100
            avg_test_mcc = np.average(np.array(test_mcc_list))
            avg_test_ll = np.average(np.array(test_ll_list))
            std_test_ll= np.std(np.array(test_ll_list), ddof=1) / np.sqrt(np.size(np.array(test_ll_list)))
            avg_acc = np.average(np.array([avg_test_acc, avg_valid_acc]))
            avg_mcc = np.average(np.array([avg_test_mcc, avg_valid_mcc]))
            avg_std_err_ll = np.average(np.array([std_test_ll, std_valid_ll]))

            perf_dict_2 = {
                        'method': [method],
                        'dataset': [dataset],
                        'avg test acc': [avg_test_acc],
                        'avg test mcc': [avg_test_mcc],
                        'avg test ll': [avg_test_ll],
                        'std error test ll': [std_test_ll],
                        'avg valid acc': [avg_valid_acc],
                        'avg valid mcc': [avg_valid_mcc],
                        'avg valid ll': [avg_valid_ll],
                        'std error valid ll': [std_valid_ll],
                        'avg acc': [avg_acc],
                        'avg mcc': [avg_mcc],
                        'avg std error ll': [avg_std_err_ll]
                    }
            df_2 = pd.DataFrame(perf_dict_2)

            if perf_df2 is None:
                perf_df2 = df_2
            else:
                perf_df2 = pd.concat([perf_df2, df_2])         

            if not os.path.exists('replication'):
                os.mkdir('replication')
            perf_df2.to_csv('replication/perf_grouped_results.csv', index = False)
            dfi.export(perf_df2,"replication/perf_grouped_results.png")

    elif args.action == 'experiment1_dropout':
        predefined_args = [
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet'
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17'
        }]
        perf_df = None
        perf_df2 = None
        state_keep_prob_arr = [1]
        dropout_iterations_arr = [[1, 1]]
        #state_keep_prob_arr = [0.5, 0.05, 0.005, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        #dropout_iterations_arr = [[16000, 19000], [2500, 500], [500, 500], [12500, 500], [12500, 500], [18000, 750], [4000, 1250], [2500, 17250], [2500, 17000], [3750, 20500], [2250, 34000], [500, 9000]]
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            method = pre['method']
            dataset = pre['dataset']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            for i, s in enumerate(state_keep_prob_arr):
                args.state_keep_prob = s
                if dataset == 'stocknet':
                    dropout_iterations = dropout_iterations_arr[i][0]
                elif dataset == 'kdd17':
                    dropout_iterations = dropout_iterations_arr[i][1]
                parameters = {
                    'seq': int(args.seq),
                    'unit': int(args.unit),
                    'alp': float(args.alpha_l2),
                    'bet': float(args.beta_adv),
                    'eps': float(args.epsilon_adv),
                    'lr': float(args.learning_rate),
                    'meth': method,
                    'data': dataset,
                    'act': args.action,
                    'state_keep_prob': args.state_keep_prob
                }
                        
                pure_LSTM = AWLSTM(
                    data_path=args.path,
                    model_path=args.model_path,
                    model_save_path=args.model_save_path,
                    parameters=parameters,
                    steps=args.step,
                    epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                    tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                    hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                    reload=args.reload,
                    dropout_activation_function = 'avg'
                )

                runs = 5
                runs_arr = [*range(runs + 1)][1:]

                perf_valid_arr = []
                perf_test_arr = []

                for r in runs_arr:
                    best_valid_perf, best_test_perf = pure_LSTM.train_monte_carlo_dropout(return_perf=True, return_pred=False, iterations=dropout_iterations, evaluate_last_epoch_only=False)
                    perf_valid_arr.append(best_valid_perf)
                    perf_test_arr.append(best_test_perf)
                    perf_dict = {
                            'method': [method],
                            'dataset': [dataset],
                            'test acc': [best_test_perf['acc'] * 100],
                            'test mcc': [best_test_perf['mcc']],
                            'test ll': [best_test_perf['ll']],
                            'valid acc': [best_valid_perf['acc'] * 100],
                            'valid mcc': [best_valid_perf['mcc']],
                            'valid ll': [best_valid_perf['ll']],
                            'dropout' : [args.state_keep_prob],
                            'run': [r]
                        }

                    df = pd.DataFrame(perf_dict)
                    if perf_df is None:
                        perf_df = df
                    else:
                        perf_df = pd.concat([perf_df, df])            

                    if not os.path.exists('experiment1'):
                        os.mkdir('experiment1')
                    perf_df.to_csv('experiment1/perf_dropout_results.csv', index = False)
                    #dfi.export(perf_df,"experiment1/perf_dropout_results.png")
                    
                valid_acc_list = list(map(lambda x: x['acc'], perf_valid_arr))
                valid_mcc_list = list(map(lambda x: x['mcc'], perf_valid_arr))
                valid_ll_list = list(map(lambda x: x['ll'], perf_valid_arr))

                avg_valid_acc = np.average(np.array(valid_acc_list)) * 100
                avg_valid_mcc = np.average(np.array(valid_mcc_list))
                avg_valid_ll = np.average(np.array(valid_ll_list))
                std_valid_ll = np.std(np.array(valid_ll_list), ddof=1) / np.sqrt(np.size(np.array(valid_ll_list)))

                test_acc_list = list(map(lambda x: x['acc'], perf_test_arr))
                test_mcc_list = list(map(lambda x: x['mcc'], perf_test_arr))
                test_ll_list = list(map(lambda x: x['ll'], perf_test_arr))

                avg_test_acc = np.average(np.array(test_acc_list)) * 100
                avg_test_mcc = np.average(np.array(test_mcc_list))
                avg_test_ll = np.average(np.array(test_ll_list))
                std_test_ll= np.std(np.array(test_ll_list), ddof=1) / np.sqrt(np.size(np.array(test_ll_list)))
                avg_acc = np.average(np.array([avg_test_acc, avg_valid_acc]))
                avg_mcc = np.average(np.array([avg_test_mcc, avg_valid_mcc]))
                avg_std_err_ll = np.average(np.array([std_test_ll, std_valid_ll]))

                perf_dict_2 = {
                            'method': [method],
                            'dataset': [dataset],
                            'dropout' : [args.state_keep_prob],
                            'avg test acc': [avg_test_acc],
                            'avg test mcc': [avg_test_mcc],
                            'avg test ll': [avg_test_ll],
                            'std error test ll': [std_test_ll],
                            'avg valid acc': [avg_valid_acc],
                            'avg valid mcc': [avg_valid_mcc],
                            'avg valid ll': [avg_valid_ll],
                            'std error valid ll': [std_valid_ll],
                            'avg acc': [avg_acc],
                            'avg mcc': [avg_mcc],
                            'avg std error ll': [avg_std_err_ll]
                        }

                df_2 = pd.DataFrame(perf_dict_2)

                if perf_df2 is None:
                    perf_df2 = df_2
                else:
                    perf_df2 = pd.concat([perf_df2, df_2])         

                if not os.path.exists('experiment1'):
                    os.mkdir('experiment1')
                perf_df2.to_csv('experiment1/perf_dropout_grouped_results.csv', index = False)
                #dfi.export(perf_df2,"experiment1/perf_dropout_grouped_results.png")
    
    elif args.action == 'experiment1_dropout_convergence':
        predefined_args = [
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet'
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17'
        }]

        perf_df = None
        perf_df2 = None
        state_keep_prob_arr = [0.5, 0.05, 0.005, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            method = pre['method']
            dataset = pre['dataset']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            for s in state_keep_prob_arr:
                args.state_keep_prob = s
                parameters = {
                    'seq': int(args.seq),
                    'unit': int(args.unit),
                    'alp': float(args.alpha_l2),
                    'bet': float(args.beta_adv),
                    'eps': float(args.epsilon_adv),
                    'lr': float(args.learning_rate),
                    'meth': method,
                    'data': dataset,
                    'act': args.action,
                    'state_keep_prob': args.state_keep_prob
                }
                        
                pure_LSTM = AWLSTM(
                    data_path=args.path,
                    model_path=args.model_path,
                    model_save_path=args.model_save_path,
                    parameters=parameters,
                    steps=args.step,
                    epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                    tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                    hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                    reload=args.reload,
                    dropout_activation_function = 'avg'
                )

                runs = 5
                runs_arr = [*range(runs + 1)][1:]   

                for r in runs_arr:
                #eg. shape(2555)
                    val_std_err_arr, test_std_err_arr = pure_LSTM.train_convergence_monte_carlo_dropout(benchmark_increment=250, max_iteration = 500000)
                    for idx, i in enumerate(val_std_err_arr):
                        val_std_err = val_std_err_arr[idx]
                        tes_std_err = test_std_err_arr[idx]
                        perf_dict = {
                                'method': [method],
                                'dataset': [dataset],
                                'dropout' : [args.state_keep_prob],
                                'val iterations convergence': [val_std_err['iterations']],
                                'tes iterations convergence': [tes_std_err['iterations']],
                                'val std of the errors': [val_std_err['avg std err']],
                                'tes std of the errors': [tes_std_err['avg std err']],
                                'confidence interval': [val_std_err['ci']],
                            }

                        df = pd.DataFrame(perf_dict)
                        if perf_df is None:
                            perf_df = df
                        else:
                            perf_df = pd.concat([perf_df, df])            

                        if not os.path.exists('experiment1'):
                            os.mkdir('experiment1')
                        perf_df.to_csv('experiment1/perf_dropout_convergence_results.csv', index = False)
                        #dfi.export(perf_df,"experiment1/perf_dropout_convergence_results.png")
                            
                    # valid_acc_list = list(map(lambda x: x['acc'], perf_valid_arr))
                    # valid_mcc_list = list(map(lambda x: x['mcc'], perf_valid_arr))
                    # valid_ll_list = list(map(lambda x: x['ll'], perf_valid_arr))

                    # avg_valid_acc = np.average(np.array(valid_acc_list)) * 100
                    # avg_valid_mcc = np.average(np.array(valid_mcc_list))
                    # avg_valid_ll = np.average(np.array(valid_ll_list))
                    # std_valid_ll = np.std(np.array(valid_ll_list), ddof=1) / np.sqrt(np.size(np.array(valid_ll_list)))

                    # test_acc_list = list(map(lambda x: x['acc'], perf_test_arr))
                    # test_mcc_list = list(map(lambda x: x['mcc'], perf_test_arr))
                    # test_ll_list = list(map(lambda x: x['ll'], perf_test_arr))

                    # avg_test_acc = np.average(np.array(test_acc_list)) * 100
                    # avg_test_mcc = np.average(np.array(test_mcc_list))
                    # avg_test_ll = np.average(np.array(test_ll_list))
                    # std_test_ll= np.std(np.array(test_ll_list), ddof=1) / np.sqrt(np.size(np.array(test_ll_list)))
                    # avg_acc = np.average(np.array([avg_test_acc, avg_valid_acc]))
                    # avg_mcc = np.average(np.array([avg_test_mcc, avg_valid_mcc]))
                    # avg_std_err_ll = np.average(np.array([std_test_ll, std_valid_ll]))

                    # perf_dict_2 = {
                    #             'method': [method],
                    #             'dataset': [dataset],
                    #             'dropout' : [args.state_keep_prob],
                    #             'avg test acc': [avg_test_acc],
                    #             'avg test mcc': [avg_test_mcc],
                    #             'avg test ll': [avg_test_ll],
                    #             'std error test ll': [std_test_ll],
                    #             'avg valid acc': [avg_valid_acc],
                    #             'avg valid mcc': [avg_valid_mcc],
                    #             'avg valid ll': [avg_valid_ll],
                    #             'std error valid ll': [std_valid_ll],
                    #             'avg acc': [avg_acc],
                    #             'avg mcc': [avg_mcc],
                    #             'avg std error ll': [avg_std_err_ll]
                    #         }

                    # df_2 = pd.DataFrame(perf_dict_2)

                    # if perf_df2 is None:
                    #     perf_df2 = df_2
                    # else:
                    #     perf_df2 = pd.concat([perf_df2, df_2])         

                    # if not os.path.exists('experiment1'):
                    #     os.mkdir('experiment1')
                    # perf_df2.to_csv('experiment1/perf_dropout_grouped_results.csv', index = False)
                    # dfi.export(perf_df2,"experiment1/perf_dropout_grouped_results.png")

    elif args.action == 'experiment2_ensemble':
        predefined_args = [  
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet',
            'param_iterations': 4
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 0,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17',
            'param_iterations': 4
        }]
        
        perf_df = None
        perf_df2 = None
        perf_df3 = None
        perf_df4 = None
        perf_ret_val_df = None
        perf_ret_tes_df = None
        prob_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            method = pre['method']
            dataset = pre['dataset']
            param_iterations = pre['param_iterations']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate),
                'meth': method,
                'data': dataset,
                'act': args.action
            }

            benchmark_df = pd.read_csv('./experiment2/replication_pre_returns_results.csv')
            best_benchmark_model = benchmark_df[(benchmark_df['dataset'] == dataset) & (benchmark_df['method'] == method)]
            best_benchmark_model = best_benchmark_model.reset_index()

            tes_best_ind = best_benchmark_model['total tes log return'].idxmax()
            tes_best_benchmark_model = best_benchmark_model.iloc[tes_best_ind]

            benchmark_val_df = pd.read_csv('./experiment2/replication_pre_val_ticker_returns_results.csv')
            best_benchmark_val_model = benchmark_val_df[(benchmark_val_df['dataset'] == tes_best_benchmark_model['dataset']) & (benchmark_val_df['method'] == tes_best_benchmark_model['method']) & (benchmark_val_df['run'] == tes_best_benchmark_model['run'])]

            benchmark_tes_df = pd.read_csv('./experiment2/replication_pre_tes_ticker_returns_results.csv')
            best_benchmark_tes_model =  benchmark_tes_df[(benchmark_tes_df['dataset'] == tes_best_benchmark_model['dataset']) & (benchmark_tes_df['method'] == tes_best_benchmark_model['method']) & (benchmark_tes_df['run'] == tes_best_benchmark_model['run'])]
 
            runs = 5
            runs_arr = [*range(runs + 1)][1:]    
            perf_valid_arr = []
            perf_test_arr = []

            pure_LSTM = AWLSTM(
                data_path=args.path,
                model_path=args.model_path,
                model_save_path=args.model_save_path,
                parameters=parameters,
                steps=args.step,
                epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                reload=args.reload,
                dropout_activation_function = None,
                load_mappings = True
            )


            for r in runs_arr:
                ensemble_model_results, best_valid_perf, best_test_perf, best_valid_pred, best_test_pred = pure_LSTM.train_ensemble(return_perf=True, return_pred=True, param_iterations=param_iterations)
                best_valid_pred = label(pure_LSTM.hinge, best_valid_pred)
                best_test_pred = label(pure_LSTM.hinge, best_test_pred)

                val_mappings = copy.copy(pure_LSTM.val_mappings)
                tes_mappings = copy.copy(pure_LSTM.tes_mappings)

                for s in prob_arr:
                    val_mappings_arr = []
                    tes_mappings_arr = []
                    for i, m in enumerate(val_mappings):
                        m['run'] = r
                        m['method'] = method
                        m['dataset'] = dataset
                        prob = best_valid_perf['prob_arr'][i]
                        m['prob'] = prob
                        m['prob_s'] = s
                        pred = best_valid_perf['pred'][i][0]
                        m['next_day_pred'] = pred
                        if prob >= s:
                            m['next_day_action'] = 0
                        elif best_valid_perf['pred'][i][0] == 0:
                            m['next_day_action'] = -1
                        elif best_valid_perf['pred'][i][0] == 1:
                            m['next_day_action'] = 1
                        prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], val_mappings))
                        if len(prev_mapping) > 0:
                            m['day_action'] = prev_mapping[0]['next_day_action']
                            m['day_action_pred'] = prev_mapping[0]['next_day_pred']
                        else:
                            m['day_action'] = 0
                            m['day_action_pred'] = 0
                        val_mappings_arr.append(m)

                    for i, m in enumerate(tes_mappings):
                        m['run'] = r
                        m['method'] = method
                        m['dataset'] = dataset
                        prob = best_test_perf['prob_arr'][i]
                        m['prob'] = prob
                        m['prob_s'] = s
                        pred = best_test_perf['pred'][i][0]
                        m['next_day_pred'] = pred
                        if prob >= s:
                            m['next_day_action'] = 0
                        elif best_test_perf['pred'][i][0] == 0:
                            m['next_day_action'] = -1
                        elif best_test_perf['pred'][i][0] == 1:
                            m['next_day_action'] = 1
                        prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], tes_mappings))
                        if len(prev_mapping) > 0:
                            m['day_action'] = prev_mapping[0]['next_day_action']
                            m['day_action_pred'] = prev_mapping[0]['next_day_pred']
                        else:
                            m['day_action'] = 0
                            m['day_action_pred'] = 0

                        tes_mappings_arr.append(m)
                
                    val_mappings_df = pd.DataFrame(val_mappings_arr)
                    tes_mappings_df = pd.DataFrame(tes_mappings_arr)

                    val_mappings_df['log_return_action'] = val_mappings_df['log_return'] * val_mappings_df['day_action']
                    val_mappings_df['log_return_action_pred'] = val_mappings_df['log_return'] * val_mappings_df['day_action_pred']
                    val_pre_returns = val_mappings_df['log_return_action'].sum()
                    
                    tes_mappings_df['log_return_action'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action']
                    tes_mappings_df['log_return_action_pred'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action_pred']
                    tes_pre_returns = tes_mappings_df['log_return_action'].sum()

                    val_pre_returns_avg = val_mappings_df['log_return_action'].mean()
                    tes_pre_returns_avg = tes_mappings_df['log_return_action'].mean()
                    val_sharp_ratio = val_pre_returns_avg / val_mappings_df['log_return_action'].std()
                    tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['log_return_action'].std()

                    val_total_trading_days_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0)].shape[0]
                    tes_total_trading_days_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0)].shape[0]
                    val_total_trading_days_successful = val_mappings_df[(val_mappings_df['log_return_action'] > 0)].shape[0]
                    tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['log_return_action'] > 0)].shape[0]
                    total_val_trading_days = val_mappings_df.shape[0]
                    total_tes_trading_days = tes_mappings_df.shape[0]
                    val_total_trading_days_correctly_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0) & (val_mappings_df['log_return_action_pred'] <= 0)].shape[0]
                    tes_total_trading_days_correctly_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0) & (tes_mappings_df['log_return_action_pred'] <= 0)].shape[0]

                    perf_dict = {
                            'method': [method],
                            'dataset': [dataset],
                            'total val trading days skipped': [val_total_trading_days_skipped],
                            'total val trading days successful': [val_total_trading_days_successful],
                            'total val trading days successfuly skipped': [val_total_trading_days_correctly_skipped],
                            'ratio of val tradiing days successfully skipped over total': [val_total_trading_days_correctly_skipped / val_total_trading_days_skipped],
                            'total val trading days':[total_val_trading_days],
                            'total val log return': [val_pre_returns],
                            'avg val log return': [val_pre_returns_avg],
                            'val sharpe ratio': [val_sharp_ratio],
                            'val acc': [best_valid_perf['acc'] * 100],
                            'val mcc': [best_valid_perf['mcc']],
                            'total tes trading days skipped': [tes_total_trading_days_skipped],
                            'total tes trading days successful': [tes_total_trading_days_successful],
                            'total tes trading days successfuly skipped': [tes_total_trading_days_correctly_skipped],
                            'ratio of tes tradiing days successfully skipped over total': [tes_total_trading_days_correctly_skipped / tes_total_trading_days_skipped],
                            'total tes trading days':[total_tes_trading_days],
                            'total tes log return': [tes_pre_returns],
                            'avg tes log return': [tes_pre_returns_avg],
                            'tes sharpe ratio': [tes_sharp_ratio],
                            'tes acc': [best_test_perf['acc'] * 100],
                            'tes mcc': [best_test_perf['mcc']],
                            'run': [r],
                            'prob': [s],
                        }
                    
                    df = pd.DataFrame(perf_dict)
                    if perf_df is None:
                        perf_df = df
                    else:
                        perf_df = pd.concat([perf_df, df])            

                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_df.to_csv('experiment2/ensemble_pre_returns_results.csv', index = False)
               
                    tickers = set(list(map(lambda x: x['ticker_filename'], val_mappings_arr)))
                    ret_val_dic = {'method': [], 'dataset': [], 'avg_prob': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'best benchmark log return': [], 'best benchmark sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': [], 'total trading days correctly skipped': [], 'std log return before action': []}
                    for t in tickers:
                        ticker_mapping = val_mappings_df[(val_mappings_df['ticker_filename'] == t)]
                        total_log_return = ticker_mapping['log_return_action'].sum()
                        avg_log_return = ticker_mapping['log_return_action'].mean()        
                        sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                        std_log_return_before_action = ticker_mapping['log_return'].std()        
                        total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                        total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                        best_benchmark_val_model_ticker = best_benchmark_val_model[(best_benchmark_val_model['ticker filename'] == t)]
                        total_trading_days = ticker_mapping.shape[0]
                        total_trading_days_correctly_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0) & (ticker_mapping['log_return_action_pred'] <= 0)].shape[0]

                        ret_val_dic['method'].append(method)
                        ret_val_dic['dataset'].append(dataset)
                        ret_val_dic['run'].append(r)  
                        ret_val_dic['ticker filename'].append(t)
                        ret_val_dic['total log return'].append(total_log_return)
                        ret_val_dic['avg log return'].append(avg_log_return)
                        ret_val_dic['sharpe ratio'].append(sharp_ratio)
                        ret_val_dic['best benchmark log return'].append(best_benchmark_val_model_ticker['total log return'].iloc[0])
                        ret_val_dic['best benchmark sharpe ratio'].append(best_benchmark_val_model_ticker['sharpe ratio'].iloc[0])
                        ret_val_dic['total trading days skipped'].append(total_trading_days_skipped)
                        ret_val_dic['total trading days successful'].append(total_trading_days_successful)
                        ret_val_dic['total trading days'].append(total_trading_days)
                        ret_val_dic['total trading days correctly skipped'].append(total_trading_days_correctly_skipped)
                        ret_val_dic['avg_prob'].append(s)
                        ret_val_dic['std log return before action'].append(std_log_return_before_action)

                    ret_val_df = pd.DataFrame(ret_val_dic)
                    if perf_ret_val_df is None:
                        perf_ret_val_df = ret_val_df
                    else:
                        perf_ret_val_df = pd.concat([perf_ret_val_df, ret_val_df])      
                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_ret_val_df.to_csv('experiment2/dropout_pre_val_ticker_returns_results.csv', index = False)

                    tickers = set(list(map(lambda x: x['ticker_filename'], tes_mappings_arr)))
                    ret_tes_dic = {'method': [], 'dataset': [], 'avg_prob': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'best benchmark log return': [], 'best benchmark sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': [], 'total trading days correctly skipped': [], 'std log return before action': []}

                    for t in tickers:
                        ticker_mapping = tes_mappings_df[(tes_mappings_df['ticker_filename'] == t)]
                        total_log_return = ticker_mapping['log_return_action'].sum()
                        avg_log_return = ticker_mapping['log_return_action'].mean()        
                        sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                        std_log_return_before_action = ticker_mapping['log_return'].std()        
                        total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                        total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                        best_benchmark_tes_model_ticker = best_benchmark_tes_model[(best_benchmark_tes_model['ticker filename'] == t)]
                        total_trading_days = ticker_mapping.shape[0]
                        total_trading_days_correctly_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0) & (ticker_mapping['log_return_action_pred'] <= 0)].shape[0]

                        ret_tes_dic['method'].append(method)
                        ret_tes_dic['dataset'].append(dataset)
                        ret_tes_dic['run'].append(r)  
                        ret_tes_dic['ticker filename'].append(t)
                        ret_tes_dic['total log return'].append(total_log_return)
                        ret_tes_dic['avg log return'].append(avg_log_return)
                        ret_tes_dic['sharpe ratio'].append(sharp_ratio)
                        ret_tes_dic['best benchmark log return'].append(best_benchmark_tes_model_ticker['total log return'].iloc[0])
                        ret_tes_dic['best benchmark sharpe ratio'].append(best_benchmark_tes_model_ticker['sharpe ratio'].iloc[0])
                        ret_tes_dic['total trading days skipped'].append(total_trading_days_skipped)
                        ret_tes_dic['total trading days successful'].append(total_trading_days_successful)
                        ret_tes_dic['total trading days'].append(total_trading_days)
                        ret_tes_dic['total trading days correctly skipped'].append(total_trading_days_correctly_skipped)
                        ret_tes_dic['avg_prob'].append(s)
                        ret_tes_dic['std log return before action'].append(std_log_return_before_action)

                    ret_tes_df = pd.DataFrame(ret_tes_dic)
                    if perf_ret_tes_df is None:
                        perf_ret_tes_df = ret_tes_df
                    else:
                        perf_ret_tes_df = pd.concat([perf_ret_tes_df, ret_tes_df])      
                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_ret_tes_df.to_csv('experiment2/ensemble_pre_tes_ticker_returns_results.csv', index = False)

                    df_3 = pd.DataFrame(val_mappings_arr)

                    if perf_df3 is None:
                        perf_df3 = df_3
                    else:
                        perf_df3 = pd.concat([perf_df3, df_3])         

                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_df3.to_csv('experiment2/ensemble_val_mapping_results.csv', index = False)

                    df_4 = pd.DataFrame(tes_mappings_arr)

                    if perf_df4 is None:
                        perf_df4 = df_4
                    else:
                        perf_df4 = pd.concat([perf_df4, df_4])         

                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_df4.to_csv('experiment2/ensemble_tes_mapping_results.csv', index = False)

            for s in prob_arr:
                avg_total_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['total val log return'].to_numpy())
                avg_total_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['total tes log return'].to_numpy())
                avg_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['avg val log return'].to_numpy())
                avg_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['avg tes log return'].to_numpy())

                avg_sharp_ratio_val = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['dataset'] ==s)]['val sharpe ratio'].to_numpy())
                avg_sharp_ratio_tes = np.average(perf_df[(perf_df['dataset'] == dataset)  & (perf_df['dataset'] ==s)]['tes sharpe ratio'].to_numpy())

                perf_dict_2 = {
                            'method': [method],
                            'dataset': [dataset],
                            'avg total val predicted return': [avg_total_val_pre_returns],
                            'avg total tes predicted return': [avg_total_tes_pre_returns],
                            'avg val predicted return': [avg_val_pre_returns],
                            'avg tes predicted return': [avg_tes_pre_returns],
                            'avg val sharpe ratio': [avg_sharp_ratio_val],
                            'avg tes sharpe ratio': [avg_sharp_ratio_tes],
                            'prob_avg': [s]              
                        }
                df_2 = pd.DataFrame(perf_dict_2)

                if perf_df2 is None:
                    perf_df2 = df_2
                else:
                    perf_df2 = pd.concat([perf_df2, df_2])         

                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_df2.to_csv('experiment2/ensemble_pre_returns_grouped_results.csv', index = False)

    elif args.action == 'experiment2_replication':
        predefined_args = [  
    {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet'
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17'
        }]
        
        perf_df = None
        perf_df2 = None
        perf_df3 = None
        perf_df4 = None
        perf_ret_val_df = None
        perf_ret_tes_df = None
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            method = pre['method']
            dataset = pre['dataset']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate),
                'meth': method,
                'data': dataset,
                'act': args.action
            }
                    
            runs = 5
            runs_arr = [*range(runs + 1)][1:]    

            pure_LSTM = AWLSTM(
                data_path=args.path,
                model_path=args.model_path,
                model_save_path=args.model_save_path,
                parameters=parameters,
                steps=args.step,
                epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                reload=args.reload,
                dropout_activation_function = None,
                load_mappings = True
            )

            for r in runs_arr:
                val_mappings_arr = []
                tes_mappings_arr = []
                best_valid_perf, best_test_perf, best_valid_pred, best_test_pred = pure_LSTM.train(return_perf=True, return_pred=True)
                best_valid_pred = label(pure_LSTM.hinge, best_valid_pred)
                best_test_pred = label(pure_LSTM.hinge, best_test_pred)

                val_mappings = copy.copy(pure_LSTM.val_mappings)
                for i, m in enumerate(val_mappings):
                    m['run'] = r
                    m['method'] = method
                    m['dataset'] = dataset
                    if best_valid_perf['pred'][i][0] == 0:
                        m['next_day_action'] = -1
                    elif best_valid_perf['pred'][i][0] == 1:
                        m['next_day_action'] = 1
                    prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], val_mappings))
                    if len(prev_mapping) > 0:
                        m['day_action'] = prev_mapping[0]['next_day_action']
                    else:
                        m['day_action'] = 0
                    val_mappings_arr.append(m)

                tes_mappings = copy.copy(pure_LSTM.tes_mappings)
                for i, m in enumerate(tes_mappings):
                    m['run'] = r
                    m['method'] = method
                    m['dataset'] = dataset
                    if best_test_perf['pred'][i][0] == 0:
                        m['next_day_action'] = -1
                    elif best_test_perf['pred'][i][0] == 1:
                        m['next_day_action'] = 1
                    prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], tes_mappings))
                    if len(prev_mapping) > 0:
                        m['day_action'] = prev_mapping[0]['next_day_action']
                    else:
                        m['day_action'] = 0

                    tes_mappings_arr.append(m)
            
                val_mappings_df = pd.DataFrame(val_mappings_arr)
                tes_mappings_df = pd.DataFrame(tes_mappings_arr)

                val_mappings_df['log_return_action'] = val_mappings_df['log_return'] * val_mappings_df['day_action']
                val_pre_returns = val_mappings_df['log_return_action'].sum()

                tes_mappings_df['log_return_action'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action']
                tes_pre_returns = tes_mappings_df['log_return_action'].sum()

                val_pre_returns_avg = val_mappings_df['log_return_action'].mean()
                tes_pre_returns_avg = tes_mappings_df['log_return_action'].mean()
                val_sharp_ratio = val_pre_returns_avg / val_mappings_df['log_return_action'].std()
                tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['log_return_action'].std()

                val_total_trading_days_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0)].shape[0]
                tes_total_trading_days_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0)].shape[0]
                val_total_trading_days_successful = val_mappings_df[(val_mappings_df['log_return_action'] > 0)].shape[0]
                tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['log_return_action'] > 0)].shape[0]
                total_val_trading_days = val_mappings_df.shape[0]
                total_tes_trading_days = tes_mappings_df.shape[0]
            
                perf_dict = {
                        'method': [method],
                        'dataset': [dataset],
                        'total val trading days skipped': [val_total_trading_days_skipped],
                        'total val trading days successful': [val_total_trading_days_successful],
                        'total val trading days': [total_val_trading_days],
                        'total val log return': [val_pre_returns],
                        'avg val log return': [val_pre_returns_avg],
                        'val sharpe ratio': [val_sharp_ratio],
                        'total tes trading days skipped': [tes_total_trading_days_skipped],
                        'total tes trading days successful': [tes_total_trading_days_successful],
                        'total tes trading days': [total_tes_trading_days],
                        'total tes log return': [tes_pre_returns],
                        'avg tes log return': [tes_pre_returns_avg],
                        'tes sharpe ratio': [tes_sharp_ratio],
                        'run': [r]
                    }

                df = pd.DataFrame(perf_dict)
                if perf_df is None:
                    perf_df = df
                else:
                    perf_df = pd.concat([perf_df, df])            

                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_df.to_csv('experiment2/replication_pre_returns_results.csv', index = False)
                #dfi.export(perf_df,"experiment2/replication_pre_returns_results.png")

                tickers = set(list(map(lambda x: x['ticker_filename'], val_mappings_arr)))

                ret_val_dic = {'method': [], 'dataset': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': []}
                for t in tickers:
                    ticker_mapping = val_mappings_df[(val_mappings_df['ticker_filename'] == t)]
                    total_log_return = ticker_mapping['log_return_action'].sum()
                    avg_log_return = ticker_mapping['log_return_action'].mean()        
                    sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                    total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                    total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                    total_trading_days = ticker_mapping.shape[0]

                    ret_val_dic['method'].append(method)
                    ret_val_dic['dataset'].append(dataset)
                    ret_val_dic['run'].append(r)  
                    ret_val_dic['ticker filename'].append(t)
                    ret_val_dic['total log return'].append(total_log_return)
                    ret_val_dic['avg log return'].append(avg_log_return)
                    ret_val_dic['sharpe ratio'].append(sharp_ratio)
                    ret_val_dic['total trading days skipped'].append(total_trading_days_skipped)
                    ret_val_dic['total trading days successful'].append(total_trading_days_successful)
                    ret_val_dic['total trading days'].append(total_trading_days)

                ret_val_df = pd.DataFrame(ret_val_dic)
                if perf_ret_val_df is None:
                    perf_ret_val_df = ret_val_df
                else:
                    perf_ret_val_df = pd.concat([perf_ret_val_df, ret_val_df])      
                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_ret_val_df.to_csv('experiment2/replication_pre_val_ticker_returns_results.csv', index = False)


                tickers = set(list(map(lambda x: x['ticker_filename'], tes_mappings_arr)))
                ret_tes_dic = {'method': [], 'dataset': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': []}

                for t in tickers:
                    ticker_mapping = tes_mappings_df[(tes_mappings_df['ticker_filename'] == t)]
                    total_log_return = ticker_mapping['log_return_action'].sum()
                    avg_log_return = ticker_mapping['log_return_action'].mean()        
                    sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                    total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                    total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                    total_trading_days = ticker_mapping.shape[0]

                    ret_tes_dic['method'].append(method)
                    ret_tes_dic['dataset'].append(dataset)
                    ret_tes_dic['run'].append(r)  
                    ret_tes_dic['ticker filename'].append(t)
                    ret_tes_dic['total log return'].append(total_log_return)
                    ret_tes_dic['avg log return'].append(avg_log_return)
                    ret_tes_dic['sharpe ratio'].append(sharp_ratio)
                    ret_tes_dic['total trading days skipped'].append(total_trading_days_skipped)
                    ret_tes_dic['total trading days successful'].append(total_trading_days_successful)
                    ret_tes_dic['total trading days'].append(total_trading_days)

                ret_tes_df = pd.DataFrame(ret_tes_dic)
                if perf_ret_tes_df is None:
                    perf_ret_tes_df = ret_tes_df
                else:
                    perf_ret_tes_df = pd.concat([perf_ret_tes_df, ret_tes_df])      
                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_ret_tes_df.to_csv('experiment2/replication_pre_tes_ticker_returns_results.csv', index = False)

            avg_total_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['total val log return'].to_numpy())
            avg_total_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) ]['total tes log return'].to_numpy())
            avg_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['avg val log return'].to_numpy())
            avg_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['avg tes log return'].to_numpy())

            avg_sharp_ratio_val = np.average(perf_df[(perf_df['dataset'] == dataset)]['val sharpe ratio'].to_numpy())
            avg_sharp_ratio_tes = np.average(perf_df[(perf_df['dataset'] == dataset)]['tes sharpe ratio'].to_numpy())

            perf_dict_2 = {
                        'method': [method],
                        'dataset': [dataset],
                        'avg total val predicted return': [avg_total_val_pre_returns],
                        'avg total tes predicted return': [avg_total_tes_pre_returns],
                        'avg val predicted return': [avg_val_pre_returns],
                        'avg tes predicted return': [avg_tes_pre_returns],
                        'avg val sharpe ratio': [avg_sharp_ratio_val],
                        'avg tes sharpe ratio': [avg_sharp_ratio_tes]              
                    }
            df_2 = pd.DataFrame(perf_dict_2)

            if perf_df2 is None:
                perf_df2 = df_2
            else:
                perf_df2 = pd.concat([perf_df2, df_2])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            perf_df2.to_csv('experiment2/replication_pre_returns_grouped_results.csv', index = False)
            #dfi.export(perf_df2,"experiment2/replication_pre_returns_grouped_results.png")

            df_3 = pd.DataFrame(val_mappings_arr)

            if perf_df3 is None:
                perf_df3 = df_3
            else:
                perf_df3 = pd.concat([perf_df3, df_3])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            perf_df3.to_csv('experiment2/replication_val_mapping_results.csv', index = False)
            #dfi.export(perf_df3,"experiment2/replication_val_mapping_results.png")


            df_4 = pd.DataFrame(tes_mappings_arr)

            if perf_df4 is None:
                perf_df4 = df_4
            else:
                perf_df4 = pd.concat([perf_df4, df_4])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            perf_df4.to_csv('experiment2/replication_tes_mapping_results.csv', index = False)
            #dfi.export(perf_df4,"experiment2/replication_tes_mapping_results.png")
    
    elif args.action == 'experiment2_replication_hybrid':
        predefined_args = [  
        {
                #-p
                'path': './data/stocknet-dataset/price/ourpped',
                #-a
                'att': 1,
                #-l
                'seq': 5,
                #-u
                'unit': 4,
                #-l2
                'alpha_l2': 1,
                #-f
                'fix_init': 0,
                #-v
                'adv': 1,
                #-rl
                'reload': 1,
                #-la
                'beta_adv': 0.01,
                #-le
                'epsilon_adv': 0.05,
                'model_path': './saved_model/acl18_alstm/exp',
                'model_save_path': './tmp/model',
                'method': 'Adv-ALSTM',
                'dataset': 'stocknet'
            },
            {
                #-p
                'path': './data/kdd17/ourpped/',
                #-a
                'att': 1,
                #-l
                'seq': 15,
                #-u
                'unit': 16,
                #-l2
                'alpha_l2': 0.001,
                #-f
                'fix_init': 1,
                #-v
                'adv': 1,
                #-rl
                'reload': 1,
                #-la
                'beta_adv': 0.05,
                #-le
                'epsilon_adv':  0.001,
                'model_path': './saved_model/kdd17_alstm/model',
                'model_save_path': './tmp/model',
                'method': 'Adv-ALSTM',
                'dataset': 'kdd17'
            }]
            
        perf_df = None
        perf_df2 = None
        perf_df3 = None
        perf_df4 = None
        perf_ret_val_df = None
        perf_ret_tes_df = None
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            method = pre['method']
            dataset = pre['dataset']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate),
                'meth': method,
                'data': dataset,
                'act': args.action
            }
                    
            benchmark_df = pd.read_csv('./experiment2/dropout_pre_returns_results.csv')
            best_benchmark_model = benchmark_df[(benchmark_df['dataset'] == dataset) & (benchmark_df['method'] == method)]
            best_benchmark_model = best_benchmark_model.reset_index()

            tes_best_ind = best_benchmark_model['total tes log return'].idxmax()
            tes_best_benchmark_model = best_benchmark_model.iloc[tes_best_ind]

            benchmark_val_df = pd.read_csv('./experiment2/dropout_val_mapping_results.csv')
            best_benchmark_val_model = benchmark_val_df[(benchmark_val_df['dataset'] == tes_best_benchmark_model['dataset']) & (benchmark_val_df['method'] == tes_best_benchmark_model['method']) & (benchmark_val_df['run'] == tes_best_benchmark_model['run'])]
            best_benchmark_val_model = best_benchmark_val_model[['ticker_filename', 'date', 'prev_date', 'log_return', 'prob', 'next_day_pred']].drop_duplicates()

            val_mappings_drop = []
            s = tes_best_benchmark_model['prob']
            for i, r in best_benchmark_val_model.iterrows():
                m = {
                    'ticker_filename': r['ticker_filename'],
                    'date': r['date'],
                    'prev_date': r['prev_date'],
                    'log_return': r['log_return'],
                    'prob': r['prob'],
                    'next_day_pred': r['next_day_pred'],
                    }
                if r['prob'] >= s:
                    m['next_day_action'] = 0
                elif r['prob'] == 0:
                    m['next_day_action'] = -1
                elif r['prob'] == 1:
                    m['next_day_action'] = 1
                
                prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], val_mappings_drop))
                if len(prev_mapping) > 0:
                    m['day_action'] = prev_mapping[0]['next_day_action']
                else:
                    m['day_action'] = 0
                
                val_mappings_drop.append(m)

            benchmark_tes_df = pd.read_csv('./experiment2/dropout_tes_mapping_results.csv')
            best_benchmark_tes_model =  benchmark_tes_df[(benchmark_tes_df['dataset'] == tes_best_benchmark_model['dataset']) & (benchmark_tes_df['method'] == tes_best_benchmark_model['method']) & (benchmark_tes_df['run'] == tes_best_benchmark_model['run'])]
            best_benchmark_tes_model = best_benchmark_val_model[['ticker_filename', 'date', 'prev_date', 'log_return', 'prob', 'next_day_pred']].drop_duplicates()
           
            tes_mappings_drop = []
            for i, r in best_benchmark_tes_model.iterrows():
                m = {
                    'ticker_filename': r['ticker_filename'],
                    'date': r['date'],
                    'prev_date': r['prev_date'],
                    'log_return': r['log_return'],
                    'prob': r['prob'],
                    'next_day_pred': r['next_day_pred'],
                    }
                
                if r['prob'] >= s:
                    m['next_day_action'] = 0
                elif r['prob'] == 0:
                    m['next_day_action'] = -1
                elif r['prob'] == 1:
                    m['next_day_action'] = 1
                
                prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], tes_mappings_drop))
                if len(prev_mapping) > 0:
                    m['day_action'] = prev_mapping[0]['next_day_action']
                else:
                    m['day_action'] = 0

                tes_mappings_drop.append(m)

            runs = 5
            runs_arr = [*range(runs + 1)][1:]    

            pure_LSTM = AWLSTM(
                data_path=args.path,
                model_path=args.model_path,
                model_save_path=args.model_save_path,
                parameters=parameters,
                steps=args.step,
                epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                reload=args.reload,
                dropout_activation_function = None,
                load_mappings = True
            )


            for r in runs_arr:
                val_mappings_arr = []
                tes_mappings_arr = []
                best_valid_perf, best_test_perf, best_valid_pred, best_test_pred = pure_LSTM.train(return_perf=True, return_pred=True)
                best_valid_pred = label(pure_LSTM.hinge, best_valid_pred)
                best_test_pred = label(pure_LSTM.hinge, best_test_pred)

                val_mappings = copy.copy(pure_LSTM.val_mappings)
                for i, m in enumerate(val_mappings):
                    m['run'] = r
                    m['method'] = method
                    m['dataset'] = dataset
                    if best_valid_perf['pred'][i][0] == 0:
                        m['next_day_action'] = -1
                    elif best_valid_perf['pred'][i][0] == 1:
                        m['next_day_action'] = 1
                    prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], val_mappings))
                    if len(prev_mapping) > 0:
                        m['day_action'] = prev_mapping[0]['next_day_action']
                    else:
                        m['day_action'] = 0
                    val_mappings_arr.append(m)

                tes_mappings = copy.copy(pure_LSTM.tes_mappings)
                for i, m in enumerate(tes_mappings):
                    m['run'] = r
                    m['method'] = method
                    m['dataset'] = dataset
                    if best_test_perf['pred'][i][0] == 0:
                        m['next_day_action'] = -1
                    elif best_test_perf['pred'][i][0] == 1:
                        m['next_day_action'] = 1
                    prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], tes_mappings))
                    if len(prev_mapping) > 0:
                        m['day_action'] = prev_mapping[0]['next_day_action']
                    else:
                        m['day_action'] = 0

                    tes_mappings_arr.append(m)
            
                val_mappings_df = pd.DataFrame(val_mappings_arr)
                tes_mappings_df = pd.DataFrame(tes_mappings_arr)

                val_mappings_df['log_return_action'] = val_mappings_df['log_return'] * val_mappings_df['day_action']
                val_pre_returns = val_mappings_df['log_return_action'].sum()

                tes_mappings_df['log_return_action'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action']
                tes_pre_returns = tes_mappings_df['log_return_action'].sum()

                val_pre_returns_avg = val_mappings_df['log_return_action'].mean()
                tes_pre_returns_avg = tes_mappings_df['log_return_action'].mean()
                val_sharp_ratio = val_pre_returns_avg / val_mappings_df['log_return_action'].std()
                tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['log_return_action'].std()

                val_total_trading_days_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0)].shape[0]
                tes_total_trading_days_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0)].shape[0]
                val_total_trading_days_successful = val_mappings_df[(val_mappings_df['log_return_action'] > 0)].shape[0]
                tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['log_return_action'] > 0)].shape[0]
                total_val_trading_days = val_mappings_df.shape[0]
                total_tes_trading_days = tes_mappings_df.shape[0]
            
                perf_dict = {
                        'method': [method],
                        'dataset': [dataset],
                        'total val trading days skipped': [val_total_trading_days_skipped],
                        'total val trading days successful': [val_total_trading_days_successful],
                        'total val trading days': [total_val_trading_days],
                        'total val log return': [val_pre_returns],
                        'avg val log return': [val_pre_returns_avg],
                        'val sharpe ratio': [val_sharp_ratio],
                        'total tes trading days skipped': [tes_total_trading_days_skipped],
                        'total tes trading days successful': [tes_total_trading_days_successful],
                        'total tes trading days': [total_tes_trading_days],
                        'total tes log return': [tes_pre_returns],
                        'avg tes log return': [tes_pre_returns_avg],
                        'tes sharpe ratio': [tes_sharp_ratio],
                        'run': [r]
                    }

                df = pd.DataFrame(perf_dict)
                if perf_df is None:
                    perf_df = df
                else:
                    perf_df = pd.concat([perf_df, df])            

                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_df.to_csv('experiment2/replication_pre_returns_results.csv', index = False)
                #dfi.export(perf_df,"experiment2/replication_pre_returns_results.png")

                tickers = set(list(map(lambda x: x['ticker_filename'], val_mappings_arr)))

                ret_val_dic = {'method': [], 'dataset': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': []}
                for t in tickers:
                    ticker_mapping = val_mappings_df[(val_mappings_df['ticker_filename'] == t)]
                    total_log_return = ticker_mapping['log_return_action'].sum()
                    avg_log_return = ticker_mapping['log_return_action'].mean()        
                    sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                    total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                    total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                    total_trading_days = ticker_mapping.shape[0]

                    ret_val_dic['method'].append(method)
                    ret_val_dic['dataset'].append(dataset)
                    ret_val_dic['run'].append(r)  
                    ret_val_dic['ticker filename'].append(t)
                    ret_val_dic['total log return'].append(total_log_return)
                    ret_val_dic['avg log return'].append(avg_log_return)
                    ret_val_dic['sharpe ratio'].append(sharp_ratio)
                    ret_val_dic['total trading days skipped'].append(total_trading_days_skipped)
                    ret_val_dic['total trading days successful'].append(total_trading_days_successful)
                    ret_val_dic['total trading days'].append(total_trading_days)

                ret_val_df = pd.DataFrame(ret_val_dic)
                if perf_ret_val_df is None:
                    perf_ret_val_df = ret_val_df
                else:
                    perf_ret_val_df = pd.concat([perf_ret_val_df, ret_val_df])      
                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_ret_val_df.to_csv('experiment2/replication_pre_val_ticker_returns_results.csv', index = False)


                tickers = set(list(map(lambda x: x['ticker_filename'], tes_mappings_arr)))
                ret_tes_dic = {'method': [], 'dataset': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': []}

                for t in tickers:
                    ticker_mapping = tes_mappings_df[(tes_mappings_df['ticker_filename'] == t)]
                    total_log_return = ticker_mapping['log_return_action'].sum()
                    avg_log_return = ticker_mapping['log_return_action'].mean()        
                    sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                    total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                    total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                    total_trading_days = ticker_mapping.shape[0]

                    ret_tes_dic['method'].append(method)
                    ret_tes_dic['dataset'].append(dataset)
                    ret_tes_dic['run'].append(r)  
                    ret_tes_dic['ticker filename'].append(t)
                    ret_tes_dic['total log return'].append(total_log_return)
                    ret_tes_dic['avg log return'].append(avg_log_return)
                    ret_tes_dic['sharpe ratio'].append(sharp_ratio)
                    ret_tes_dic['total trading days skipped'].append(total_trading_days_skipped)
                    ret_tes_dic['total trading days successful'].append(total_trading_days_successful)
                    ret_tes_dic['total trading days'].append(total_trading_days)

                ret_tes_df = pd.DataFrame(ret_tes_dic)
                if perf_ret_tes_df is None:
                    perf_ret_tes_df = ret_tes_df
                else:
                    perf_ret_tes_df = pd.concat([perf_ret_tes_df, ret_tes_df])      
                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_ret_tes_df.to_csv('experiment2/replication_pre_tes_ticker_returns_results.csv', index = False)

            avg_total_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['total val log return'].to_numpy())
            avg_total_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) ]['total tes log return'].to_numpy())
            avg_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['avg val log return'].to_numpy())
            avg_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['avg tes log return'].to_numpy())

            avg_sharp_ratio_val = np.average(perf_df[(perf_df['dataset'] == dataset)]['val sharpe ratio'].to_numpy())
            avg_sharp_ratio_tes = np.average(perf_df[(perf_df['dataset'] == dataset)]['tes sharpe ratio'].to_numpy())

            perf_dict_2 = {
                        'method': [method],
                        'dataset': [dataset],
                        'avg total val predicted return': [avg_total_val_pre_returns],
                        'avg total tes predicted return': [avg_total_tes_pre_returns],
                        'avg val predicted return': [avg_val_pre_returns],
                        'avg tes predicted return': [avg_tes_pre_returns],
                        'avg val sharpe ratio': [avg_sharp_ratio_val],
                        'avg tes sharpe ratio': [avg_sharp_ratio_tes]              
                    }
            df_2 = pd.DataFrame(perf_dict_2)

            if perf_df2 is None:
                perf_df2 = df_2
            else:
                perf_df2 = pd.concat([perf_df2, df_2])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            perf_df2.to_csv('experiment2/replication_pre_returns_grouped_results.csv', index = False)
            #dfi.export(perf_df2,"experiment2/replication_pre_returns_grouped_results.png")

            df_3 = pd.DataFrame(val_mappings_arr)

            if perf_df3 is None:
                perf_df3 = df_3
            else:
                perf_df3 = pd.concat([perf_df3, df_3])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            perf_df3.to_csv('experiment2/replication_val_mapping_results.csv', index = False)
            #dfi.export(perf_df3,"experiment2/replication_val_mapping_results.png")


            df_4 = pd.DataFrame(tes_mappings_arr)

            if perf_df4 is None:
                perf_df4 = df_4
            else:
                perf_df4 = pd.concat([perf_df4, df_4])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            perf_df4.to_csv('experiment2/replication_tes_mapping_results.csv', index = False)
            #dfi.export(perf_df4,"experiment2/replication_tes_mapping_results.png")

    elif args.action == 'experiment1_ensemble':
        predefined_args = [
    {
        #-p
        'path': './data/stocknet-dataset/price/ourpped',
        #-a
        'att': 1,
        #-l
        'seq': 5,
        #-u
        'unit': 4,
        #-l2
        'alpha_l2': 1,
        #-f
        'fix_init': 0,
        #-v
        'adv': 1,
        #-rl
        'reload': 1,
        #-la
        'beta_adv': 0.01,
        #-le
        'epsilon_adv': 0.05,
        'model_path': './saved_model/acl18_alstm/exp',
        'model_save_path': './tmp/model',
        'method': 'Adv-ALSTM',
        'dataset': 'stocknet'
    },
    {
        #-p
        'path': './data/kdd17/ourpped/',
        #-a
        'att': 1,
        #-l
        'seq': 15,
        #-u
        'unit': 16,
        #-l2
        'alpha_l2': 0.001,
        #-f
        'fix_init': 1,
        #-v
        'adv': 1,
        #-rl
        'reload': 1,
        #-la
        'beta_adv': 0.05,
        #-le
        'epsilon_adv':  0.001,
        'model_path': './saved_model/kdd17_alstm/model',
        'model_save_path': './tmp/model',
        'method': 'Adv-ALSTM',
        'dataset': 'kdd17'
    }]

        perf_df = None
        perf_df2 = None
        perf_df3 = None
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            method = pre['method']
            dataset = pre['dataset']

            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            
            ensemble_parameters = [
                {
                    "param_iterations": 5,
                    "seq": 20
                },
                {
                    "param_iterations": 4,
                    "seq": 20
                },
                {
                    "param_iterations": 3,
                    "seq": 20
                } 
            ]

            for e in ensemble_parameters:
                runs = 5
                runs_arr = [*range(runs + 1)][1:]    
                perf_valid_arr = []
                perf_test_arr = []
                ensemble_model_results = []
                param_iterations = e['param_iterations']
                parameters = {
                    'seq': int(args.seq),
                    'unit': int(args.unit),
                    'alp': float(args.alpha_l2),
                    'bet': float(args.beta_adv),
                    'eps': float(args.epsilon_adv),
                    'lr': float(args.learning_rate),
                    'meth': method,
                    'data': dataset,
                    'act': args.action,
                    'seq': int(e['seq'])
                }
            
                pure_LSTM = AWLSTM(
                        data_path=args.path,
                        model_path=args.model_path,
                        model_save_path=args.model_save_path,
                        parameters=parameters,
                        steps=args.step,
                        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                        tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                        hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                        reload=args.reload,
                        dropout_activation_function = None
                )
                for r in runs_arr:

                    ensemble_model_results, best_valid_perf, best_test_perf = pure_LSTM.train_ensemble(return_perf=True, return_pred=False, param_iterations=param_iterations)
                    perf_valid_arr.append(best_valid_perf)
                    perf_test_arr.append(best_test_perf)
                
                    perf_dict = {
                            'method': [method],
                            'dataset': [dataset],
                            'test acc': [best_test_perf['acc'] * 100],
                            'test mcc': [best_test_perf['mcc']],
                            'test ll': [best_test_perf['ll']],
                            'valid acc': [best_valid_perf['acc'] * 100],
                            'valid mcc': [best_valid_perf['mcc']],
                            'valid ll': [best_valid_perf['ll']],
                            'ensemble models': [ensemble_model_results.size],
                            'param iterations': [param_iterations],
                            'run': [r]
                        }

                    df = pd.DataFrame(perf_dict)
                    if perf_df is None:
                        perf_df = df
                    else:
                        perf_df = pd.concat([perf_df, df])            

                    if not os.path.exists('experiment1'):
                        os.mkdir('experiment1')
                    perf_df.to_csv('experiment1/perf_ensemble_results.csv', index = False)
                    dfi.export(perf_df,"experiment1/perf_ensemble_results.png")

                valid_acc_list = list(map(lambda x: x['acc'], perf_valid_arr))
                valid_mcc_list = list(map(lambda x: x['mcc'], perf_valid_arr))
                valid_ll_list = list(map(lambda x: x['ll'], perf_valid_arr))

                avg_valid_acc = np.average(np.array(valid_acc_list)) * 100
                avg_valid_mcc = np.average(np.array(valid_mcc_list))
                avg_valid_ll = np.average(np.array(valid_ll_list))
                std_valid_ll = np.std(np.array(valid_ll_list), ddof=1) / np.sqrt(np.size(np.array(valid_ll_list)))

                test_acc_list = list(map(lambda x: x['acc'], perf_test_arr))
                test_mcc_list = list(map(lambda x: x['mcc'], perf_test_arr))
                test_ll_list = list(map(lambda x: x['ll'], perf_test_arr))

                avg_test_acc = np.average(np.array(test_acc_list)) * 100
                avg_test_mcc = np.average(np.array(test_mcc_list))
                avg_test_ll = np.average(np.array(test_ll_list))
                std_test_ll= np.std(np.array(test_ll_list), ddof=1) / np.sqrt(np.size(np.array(test_ll_list)))
                avg_acc = np.average(np.array([avg_test_acc, avg_valid_acc]))
                avg_mcc = np.average(np.array([avg_test_mcc, avg_valid_mcc]))
                avg_std_err_ll = np.average(np.array([std_test_ll, std_valid_ll]))

                perf_dict_2 = {
                            'method': [method],
                            'dataset': [dataset],
                            'param iterations': [param_iterations],
                            'avg test acc': [avg_test_acc],
                            'avg test mcc': [avg_test_mcc],
                            'avg test ll': [avg_test_ll],
                            'std error test ll': [std_test_ll],
                            'avg valid acc': [avg_valid_acc],
                            'avg valid mcc': [avg_valid_mcc],
                            'avg valid ll': [avg_valid_ll],
                            'std error valid ll': [std_valid_ll],
                            'avg acc': [avg_acc],
                            'avg mcc': [avg_mcc],
                            'avg std error ll': [avg_std_err_ll]
                        }
                df_2 = pd.DataFrame(perf_dict_2)

                if perf_df2 is None:
                    perf_df2 = df_2
                else:
                    perf_df2 = pd.concat([perf_df2, df_2])         

                if not os.path.exists('experiment1'):
                    os.mkdir('experiment1')
                perf_df2.to_csv('experiment1/perf_ensemble_grouped_results.csv', index = False)
                dfi.export(perf_df2,"experiment1/perf_ensemble_grouped_results.png")

                feat_dim_arr = []
                seq_arr = []
                param_iterations_arr = []

                for e in ensemble_model_results:
                    feat_dim_arr.append(e['feat_dim'])
                    seq_arr.append(e['seq'])

                method_arr = []
                dataset_arr = []
                for l in range(len(seq_arr)):
                    method_arr.append(method)
                    dataset_arr.append(dataset)
                    param_iterations_arr.append(param_iterations)

                perf_dict_3 = {
                            'method': method_arr,
                            'dataset': dataset_arr,
                            'param iterations': param_iterations_arr,
                            'seq': seq_arr,
                            'feat_dim': feat_dim_arr
                        }
                df_3 = pd.DataFrame(perf_dict_3)
                
                if perf_df3 is None:
                    perf_df3 = df_3
                else:
                    perf_df3 = pd.concat([perf_df3, df_3])            


                if not os.path.exists('experiment1'):
                    os.mkdir('experiment1')
                df_3.to_csv('experiment1/perf_ensemble_parameters_used.csv', index = False)
                dfi.export(df_3,"experiment1/perf_ensemble_parameters_used.png")
 
    elif args.action == 'experiment2_dropout':
        predefined_args = [
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet',
            'dropout_iterations': 2250,
            'state_keep_prob': 0.85
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17', 
            'dropout_iterations': 500,
            'state_keep_prob': 0.05
        }]
        
        perf_df = None
        perf_df2 = None
        perf_df3 = None
        perf_df4 = None
        perf_ret_val_df = None
        perf_ret_tes_df = None
        prob_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for pre in predefined_args:
            args.path = pre['path']
            args.att = pre['att']
            args.seq = pre['seq']
            args.unit = pre['unit']
            args.alpha_l2 = pre['alpha_l2']
            args.fix_init = pre['fix_init']
            args.adv = pre['adv']
            args.reload = pre['reload']
            args.beta_adv = pre['beta_adv']
            args.epsilon_adv = pre['epsilon_adv']
            args.model_path = pre['model_path']
            args.state_keep_prob = pre['state_keep_prob']
            method = pre['method']
            dataset = pre['dataset']
            if dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected path: %s' % args.path)
                exit(0)

            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate),
                'meth': method,
                'data': dataset,
                'act': args.action,
                'state_keep_prob': args.state_keep_prob
            }

            benchmark_df = pd.read_csv('./experiment2/replication_pre_returns_results.csv')
            best_benchmark_model = benchmark_df[(benchmark_df['dataset'] == dataset) & (benchmark_df['method'] == method)]
            best_benchmark_model = best_benchmark_model.reset_index()

            tes_best_ind = best_benchmark_model['total tes log return'].idxmax()
            tes_best_benchmark_model = best_benchmark_model.iloc[tes_best_ind]

            benchmark_val_df = pd.read_csv('./experiment2/replication_pre_val_ticker_returns_results.csv')
            best_benchmark_val_model = benchmark_val_df[(benchmark_val_df['dataset'] == tes_best_benchmark_model['dataset']) & (benchmark_val_df['method'] == tes_best_benchmark_model['method']) & (benchmark_val_df['run'] == tes_best_benchmark_model['run'])]

            benchmark_tes_df = pd.read_csv('./experiment2/replication_pre_tes_ticker_returns_results.csv')
            best_benchmark_tes_model =  benchmark_tes_df[(benchmark_tes_df['dataset'] == tes_best_benchmark_model['dataset']) & (benchmark_tes_df['method'] == tes_best_benchmark_model['method']) & (benchmark_tes_df['run'] == tes_best_benchmark_model['run'])]

            runs = 5
            runs_arr = [*range(runs + 1)][1:]
            perf_valid_arr = []
            perf_test_arr = []

            pure_LSTM = AWLSTM(
                data_path=args.path,
                model_path=args.model_path,
                model_save_path=args.model_save_path,
                parameters=parameters,
                steps=args.step,
                epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
                hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                reload=args.reload,
                dropout_activation_function = 'avg',
                load_mappings = True
            )

            for r in runs_arr:

                best_valid_perf, best_test_perf, best_valid_pred, best_test_pred = pure_LSTM.train_monte_carlo_dropout(return_perf=True, return_pred=True, iterations=pre['dropout_iterations'])
                best_valid_pred = label(pure_LSTM.hinge, best_valid_pred)
                best_test_pred = label(pure_LSTM.hinge, best_test_pred)

                val_mappings = copy.copy(pure_LSTM.val_mappings)
                tes_mappings = copy.copy(pure_LSTM.tes_mappings)

                for s in prob_arr:
                    val_mappings_arr = []
                    tes_mappings_arr = []
                    for i, m in enumerate(val_mappings):
                        m['run'] = r
                        m['method'] = method
                        m['dataset'] = dataset
                        prob = best_valid_perf['prob_arr'][i]
                        m['prob'] = prob
                        m['prob_s'] = s
                        pred = best_valid_perf['pred'][i][0]
                        m['next_day_pred'] = pred
                        if prob >= s:
                            m['next_day_action'] = 0
                        elif best_valid_perf['pred'][i][0] == 0:
                            m['next_day_action'] = -1
                        elif best_valid_perf['pred'][i][0] == 1:
                            m['next_day_action'] = 1
                        prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], val_mappings))
                        if len(prev_mapping) > 0:
                            m['day_action'] = prev_mapping[0]['next_day_action']
                            m['day_action_pred'] = prev_mapping[0]['next_day_pred']
                        else:
                            m['day_action'] = 0
                            m['day_action_pred'] = 0
                        val_mappings_arr.append(m)

                    for i, m in enumerate(tes_mappings):
                        m['run'] = r
                        m['method'] = method
                        m['dataset'] = dataset
                        prob = best_test_perf['prob_arr'][i]
                        m['prob'] = prob
                        m['prob_s'] = s
                        pred = best_test_perf['pred'][i][0]
                        m['next_day_pred'] = pred
                        if prob >= s:
                            m['next_day_action'] = 0
                        elif best_test_perf['pred'][i][0] == 0:
                            m['next_day_action'] = -1
                        elif best_test_perf['pred'][i][0] == 1:
                            m['next_day_action'] = 1
                        prev_mapping = list(filter(lambda x: x['date'] == m['prev_date'] and x['ticker_filename'] == m['ticker_filename'], tes_mappings))
                        if len(prev_mapping) > 0:
                            m['day_action'] = prev_mapping[0]['next_day_action']
                            m['day_action_pred'] = prev_mapping[0]['next_day_pred']
                        else:
                            m['day_action'] = 0
                            m['day_action_pred'] = 0

                        tes_mappings_arr.append(m)
                
                    val_mappings_df = pd.DataFrame(val_mappings_arr)
                    tes_mappings_df = pd.DataFrame(tes_mappings_arr)

                    val_mappings_df['log_return_action'] = val_mappings_df['log_return'] * val_mappings_df['day_action']
                    val_mappings_df['log_return_action_pred'] = val_mappings_df['log_return'] * val_mappings_df['day_action_pred']
                    val_pre_returns = val_mappings_df['log_return_action'].sum()
                    
                    tes_mappings_df['log_return_action'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action']
                    tes_mappings_df['log_return_action_pred'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action_pred']
                    tes_pre_returns = tes_mappings_df['log_return_action'].sum()

                    val_pre_returns_avg = val_mappings_df['log_return_action'].mean()
                    tes_pre_returns_avg = tes_mappings_df['log_return_action'].mean()
                    val_sharp_ratio = val_pre_returns_avg / val_mappings_df['log_return_action'].std()
                    tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['log_return_action'].std()

                    val_total_trading_days_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0)].shape[0]
                    tes_total_trading_days_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0)].shape[0]
                    val_total_trading_days_successful = val_mappings_df[(val_mappings_df['log_return_action'] > 0)].shape[0]
                    tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['log_return_action'] > 0)].shape[0]
                    total_val_trading_days = val_mappings_df.shape[0]
                    total_tes_trading_days = tes_mappings_df.shape[0]
                    val_total_trading_days_correctly_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0) & (val_mappings_df['log_return_action_pred'] <= 0)].shape[0]
                    tes_total_trading_days_correctly_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0) & (tes_mappings_df['log_return_action_pred'] <= 0)].shape[0]

                    perf_dict = {
                            'method': [method],
                            'dataset': [dataset],
                            'total val trading days skipped': [val_total_trading_days_skipped],
                            'total val trading days successful': [val_total_trading_days_successful],
                            'total val trading days successfuly skipped': [val_total_trading_days_correctly_skipped],
                            'ratio of val tradiing days successfully skipped over total': [val_total_trading_days_correctly_skipped / val_total_trading_days_skipped],
                            'total val trading days':[total_val_trading_days],
                            'total val log return': [val_pre_returns],
                            'avg val log return': [val_pre_returns_avg],
                            'val sharpe ratio': [val_sharp_ratio],
                            'val acc': [best_valid_perf['acc'] * 100],
                            'val mcc': [best_valid_perf['mcc']],
                            'total tes trading days skipped': [tes_total_trading_days_skipped],
                            'total tes trading days successful': [tes_total_trading_days_successful],
                            'total tes trading days successfuly skipped': [tes_total_trading_days_correctly_skipped],
                            'ratio of tes tradiing days successfully skipped over total': [tes_total_trading_days_correctly_skipped / tes_total_trading_days_skipped],
                            'total tes trading days':[total_tes_trading_days],
                            'total tes log return': [tes_pre_returns],
                            'avg tes log return': [tes_pre_returns_avg],
                            'tes sharpe ratio': [tes_sharp_ratio],
                            'tes acc': [best_test_perf['acc'] * 100],
                            'tes mcc': [best_test_perf['mcc']],
                            'run': [r],
                            'prob': [s],
                        }
                    
                    df = pd.DataFrame(perf_dict)
                    if perf_df is None:
                        perf_df = df
                    else:
                        perf_df = pd.concat([perf_df, df])            

                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_df.to_csv('experiment2/dropout_pre_returns_results.csv', index = False)
                    #dfi.export(perf_df,"experiment2/dropout_pre_returns_results.png")
               
                    tickers = set(list(map(lambda x: x['ticker_filename'], val_mappings_arr)))
                    ret_val_dic = {'method': [], 'dataset': [], 'avg_prob': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'best benchmark log return': [], 'best benchmark sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': [], 'total trading days correctly skipped': [], 'std log return before action': []}
                    for t in tickers:
                        ticker_mapping = val_mappings_df[(val_mappings_df['ticker_filename'] == t)]
                        total_log_return = ticker_mapping['log_return_action'].sum()
                        avg_log_return = ticker_mapping['log_return_action'].mean()        
                        sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                        std_log_return_before_action = ticker_mapping['log_return'].std()        
                        total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                        total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                        best_benchmark_val_model_ticker = best_benchmark_val_model[(best_benchmark_val_model['ticker filename'] == t)]
                        total_trading_days = ticker_mapping.shape[0]
                        total_trading_days_correctly_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0) & (ticker_mapping['log_return_action_pred'] <= 0)].shape[0]

                        ret_val_dic['method'].append(method)
                        ret_val_dic['dataset'].append(dataset)
                        ret_val_dic['run'].append(r)  
                        ret_val_dic['ticker filename'].append(t)
                        ret_val_dic['total log return'].append(total_log_return)
                        ret_val_dic['avg log return'].append(avg_log_return)
                        ret_val_dic['sharpe ratio'].append(sharp_ratio)
                        ret_val_dic['best benchmark log return'].append(best_benchmark_val_model_ticker['total log return'].iloc[0])
                        ret_val_dic['best benchmark sharpe ratio'].append(best_benchmark_val_model_ticker['sharpe ratio'].iloc[0])
                        ret_val_dic['total trading days skipped'].append(total_trading_days_skipped)
                        ret_val_dic['total trading days successful'].append(total_trading_days_successful)
                        ret_val_dic['total trading days'].append(total_trading_days)
                        ret_val_dic['total trading days correctly skipped'].append(total_trading_days_correctly_skipped)
                        ret_val_dic['avg_prob'].append(s)
                        ret_val_dic['std log return before action'].append(std_log_return_before_action)

                    ret_val_df = pd.DataFrame(ret_val_dic)
                    if perf_ret_val_df is None:
                        perf_ret_val_df = ret_val_df
                    else:
                        perf_ret_val_df = pd.concat([perf_ret_val_df, ret_val_df])      
                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_ret_val_df.to_csv('experiment2/dropout_pre_val_ticker_returns_results.csv', index = False)

                    tickers = set(list(map(lambda x: x['ticker_filename'], tes_mappings_arr)))
                    ret_tes_dic = {'method': [], 'dataset': [], 'avg_prob': [], 'run': [], 'ticker filename': [], 'total log return': [], 'avg log return': [], 'sharpe ratio': [], 'best benchmark log return': [], 'best benchmark sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': [], 'total trading days correctly skipped': [], 'std log return before action': []}

                    for t in tickers:
                        ticker_mapping = tes_mappings_df[(tes_mappings_df['ticker_filename'] == t)]
                        total_log_return = ticker_mapping['log_return_action'].sum()
                        avg_log_return = ticker_mapping['log_return_action'].mean()        
                        sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                        std_log_return_before_action = ticker_mapping['log_return'].std()        
                        total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                        total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                        best_benchmark_tes_model_ticker = best_benchmark_tes_model[(best_benchmark_tes_model['ticker filename'] == t)]
                        total_trading_days = ticker_mapping.shape[0]
                        total_trading_days_correctly_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0) & (ticker_mapping['log_return_action_pred'] <= 0)].shape[0]

                        ret_tes_dic['method'].append(method)
                        ret_tes_dic['dataset'].append(dataset)
                        ret_tes_dic['run'].append(r)  
                        ret_tes_dic['ticker filename'].append(t)
                        ret_tes_dic['total log return'].append(total_log_return)
                        ret_tes_dic['avg log return'].append(avg_log_return)
                        ret_tes_dic['sharpe ratio'].append(sharp_ratio)
                        ret_tes_dic['best benchmark log return'].append(best_benchmark_tes_model_ticker['total log return'].iloc[0])
                        ret_tes_dic['best benchmark sharpe ratio'].append(best_benchmark_tes_model_ticker['sharpe ratio'].iloc[0])
                        ret_tes_dic['total trading days skipped'].append(total_trading_days_skipped)
                        ret_tes_dic['total trading days successful'].append(total_trading_days_successful)
                        ret_tes_dic['total trading days'].append(total_trading_days)
                        ret_tes_dic['total trading days correctly skipped'].append(total_trading_days_correctly_skipped)
                        ret_tes_dic['avg_prob'].append(s)
                        ret_tes_dic['std log return before action'].append(std_log_return_before_action)

                    ret_tes_df = pd.DataFrame(ret_tes_dic)
                    if perf_ret_tes_df is None:
                        perf_ret_tes_df = ret_tes_df
                    else:
                        perf_ret_tes_df = pd.concat([perf_ret_tes_df, ret_tes_df])      
                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_ret_tes_df.to_csv('experiment2/dropout_pre_tes_ticker_returns_results.csv', index = False)

                    df_3 = pd.DataFrame(val_mappings_arr)

                    if perf_df3 is None:
                        perf_df3 = df_3
                    else:
                        perf_df3 = pd.concat([perf_df3, df_3])         

                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_df3.to_csv('experiment2/dropout_val_mapping_results.csv', index = False)

                    df_4 = pd.DataFrame(tes_mappings_arr)

                    if perf_df4 is None:
                        perf_df4 = df_4
                    else:
                        perf_df4 = pd.concat([perf_df4, df_4])         

                    if not os.path.exists('experiment2'):
                        os.mkdir('experiment2')
                    perf_df4.to_csv('experiment2/dropout_tes_mapping_results.csv', index = False)

            for s in prob_arr:
                avg_total_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['total val log return'].to_numpy())
                avg_total_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['total tes log return'].to_numpy())
                avg_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['avg val log return'].to_numpy())
                avg_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['prob'] == s)]['avg tes log return'].to_numpy())

                avg_sharp_ratio_val = np.average(perf_df[(perf_df['dataset'] == dataset) & (perf_df['dataset'] ==s)]['val sharpe ratio'].to_numpy())
                avg_sharp_ratio_tes = np.average(perf_df[(perf_df['dataset'] == dataset)  & (perf_df['dataset'] ==s)]['tes sharpe ratio'].to_numpy())

                perf_dict_2 = {
                            'method': [method],
                            'dataset': [dataset],
                            'avg total val predicted return': [avg_total_val_pre_returns],
                            'avg total tes predicted return': [avg_total_tes_pre_returns],
                            'avg val predicted return': [avg_val_pre_returns],
                            'avg tes predicted return': [avg_tes_pre_returns],
                            'avg val sharpe ratio': [avg_sharp_ratio_val],
                            'avg tes sharpe ratio': [avg_sharp_ratio_tes],
                            'prob_avg': [s]              
                        }
                df_2 = pd.DataFrame(perf_dict_2)

                if perf_df2 is None:
                    perf_df2 = df_2
                else:
                    perf_df2 = pd.concat([perf_df2, df_2])         

                if not os.path.exists('experiment2'):
                    os.mkdir('experiment2')
                perf_df2.to_csv('experiment2/dropout_pre_returns_grouped_results.csv', index = False)
                #dfi.export(perf_df2,"experiment2/dropout_pre_returns_grouped_results.png")
    else:
        print(args)

        dropout_activation_function = None
        if (args.state_keep_prob < 1):
            dropout_activation_function = 'std'

        if dataset == 'stocknet':
            tra_date = '2014-01-02'
            val_date = '2015-08-03'
            tes_date = '2015-10-01'
        elif dataset == 'kdd17':
            tra_date = '2007-01-03'
            val_date = '2015-01-02'
            tes_date = '2016-01-04'
        else:
            print('unexpected path: %s' % args.path)
            exit(0)
        parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate),
                'meth': method,
                'data': dataset,
                'act': args.action,
                'state_keep_prob': args.state_keep_prob
            }
                    
        pure_LSTM = AWLSTM(
            data_path=args.path,
            model_path=args.model_path,
            model_save_path=args.model_save_path,
            parameters=parameters,
            steps=args.step,
            epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
            tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
            hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
            reload=args.reload,
            dropout_activation_function = dropout_activation_function
        )
        if args.action == 'train':
                pure_LSTM.train()      
        elif args.action == 'test':
            pure_LSTM.test()
        elif args.action == 'pred':
            pure_LSTM.predict_record()
        elif args.action == 'adv':
            pure_LSTM.predict_adv()
        elif args.action == 'latent':
            pure_LSTM.get_latent_rep()