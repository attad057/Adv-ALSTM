import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time
from scipy.stats import entropy
import pickle     
#from pred_lstm_experiment_2 import experiment2_dropout               
import threading
import queue
import math

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
        pre_std = np.std(pre_arr_s, axis = 1)
        pre_var = np.var(pre_arr_s, axis = 1)
        pre_entr = np.zeros((pre_std.shape[0]))

        for i, x in enumerate(pre_arr_s):
            value, counts = np.unique(x, return_counts=True)
            entr = entropy(counts, base=None)
            pre_entr[i] = entr
        #pre_avg_round = np.reshape(np.round(pre_avg), (pre_np_arr.shape[1], pre_np_arr.shape[2]))

        #take best indexes from pre_np_arr
        # ind_rows = np.arange(pre_np_arr.shape[1])
        # pre = np.reshape(pre_arr_s[ind_rows, ind_columns], (pre_np_arr.shape[1], pre_np_arr.shape[2]))
        #threshold 0.5 and up = 1 else 0
        return np.reshape(np.floor(pre_avg + 0.5), (pre_np_arr.shape[1], pre_np_arr.shape[2])), [{'measure': 'std', 'val': pre_std},{'measure': 'var', 'val': pre_var},{'measure': 'entr', 'val': pre_entr}]

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

    def evaluate_monte_carlo(self, iterations, isValidation, sess):
        result_queue = queue.Queue()
        def run_session(fetches, feed_dict, hinge):
            loss, pre = sess.run(
                fetches, feed_dict
            )
            result_queue.put((label(hinge, pre), loss))
        num_threads = iterations
        iterations_arr = [*range(int(math.ceil(iterations / num_threads)) + 1)][1:]

        if sess is None:
            self.construct_graph()
            sess = tf.Session()
            saver = tf.train.Saver()
            if self.reload:
                saver.restore(sess, self.model_save_path)
                print('model restored')
            else:
                sess.run(tf.global_variables_initializer())

        if isValidation:
            feed_dict = {
                    self.pv_var: self.val_pv,
                    self.wd_var: self.val_wd,
                    self.gt_var: self.val_gt,
                    self.state_keep_prob_var: self.state_keep_prob,
                    self.input_keep_prob_var: self.input_keep_prob,
                    self.output_keep_prob_var: self.output_keep_prob
                }
            gt = self.val_gt
        else:
            feed_dict = {
                    self.pv_var: self.tes_pv,
                    self.wd_var: self.tes_wd,
                    self.gt_var: self.tes_gt,
                    self.state_keep_prob_var: self.state_keep_prob,
                    self.input_keep_prob_var: self.input_keep_prob,
                    self.output_keep_prob_var: self.output_keep_prob
                }
            gt = self.tes_gt
        l_arr = [] 
        threads = []
        for r in iterations_arr:
            for _ in range(num_threads):
                thread = threading.Thread(target=run_session, args=((self.loss, self.pred), feed_dict, self.hinge))
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            while not result_queue.empty():
                result = result_queue.get()
                l_arr.append(result[0])
                loss = result[1]

        #shape (3720, 10)
        l_np_arr = np.array(l_arr)

        if self.dropout_activation_function == 'avg':
            pre, pre_prob = self.monte_carlo_average(l_np_arr)

        cur_perf = evaluate(pre, gt, self.hinge, additional_metrics=True)
        cur_perf['prob_arr'] = pre_prob

        cur_perf_p = {
            'acc': cur_perf['acc'],
            'mcc': cur_perf['mcc']
        }        
        if isValidation:
            print('\Val per:', cur_perf_p, '\Val loss:', loss, '\vVal state_keep_prob:', self.state_keep_prob)
        else:
            print('\tTest per:', cur_perf_p, '\tTest loss:', loss, '\Test state_keep_prob:', self.state_keep_prob)

        if sess is None:
            sess.close()
            tf.reset_default_graph()

        return cur_perf, pre

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

    def train_monte_carlo_dropout(self, tune_para=False, return_perf=False, return_pred=True, iterations_arr=[1]):
        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        best_test_perf = {
            'acc': 0, 'mcc': -2
        }
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.intra_op_parallelism_threads = 5
        # config.inter_op_parallelism_threads = 5
        self.construct_graph()
        #sess = tf.Session(config = config)
        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)
        best_iterations = 0

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
            feed_dict = {
                self.pv_var: self.val_pv,
                self.wd_var: self.val_wd,
                self.gt_var: self.val_gt,
                self.state_keep_prob_var: self.state_keep_prob,
                self.input_keep_prob_var: self.input_keep_prob,
                self.output_keep_prob_var: self.output_keep_prob
            }

            val_loss, val_pre = sess.run(
                    (self.loss, self.pred), feed_dict
                )
            cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge, additional_metrics=True)
            cur_valid_perf_p = {
                'acc': cur_valid_perf['acc'],
                'mcc': cur_valid_perf['mcc']
            }
            print('\tVal per:', cur_valid_perf_p, '\tVal loss:', val_loss, '\tVal state_keep_prob:', self.state_keep_prob)

            # test on test set
            feed_dict = {
                self.pv_var: self.tes_pv,
                self.wd_var: self.tes_wd,
                self.gt_var: self.tes_gt,
                self.state_keep_prob_var: self.state_keep_prob,
                self.input_keep_prob_var: self.input_keep_prob,
                self.output_keep_prob_var: self.output_keep_prob
            }

            test_loss, tes_pre = sess.run(
                    (self.loss, self.pred), feed_dict
                )
            cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge, additional_metrics=True)
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
            
        sess.close()
        tf.reset_default_graph()

        best_valid_perf_dropout = None
        best_test_perf_dropout = None
        best_val_pre = None
        best_test_pre = None

        for i2, d in enumerate(iterations_arr):
            curr_valid_perf_dropout, curr_val_pre = self.evaluate_monte_carlo(d, True, None)
            curr_test_perf_dropout, curr_tes_pre = self.evaluate_monte_carlo(d, False, None)

            if best_valid_perf_dropout is None or curr_valid_perf_dropout['acc'] > best_valid_perf_dropout['acc']:
                best_valid_perf_dropout = copy.copy(curr_valid_perf_dropout)
                best_test_perf_dropout = copy.copy(curr_test_perf_dropout)
                best_val_pre = copy.copy(curr_val_pre)
                best_test_pre = copy.copy(curr_tes_pre)
                best_iterations = d

        best_valid_perf = best_valid_perf_dropout
        best_test_perf = best_test_perf_dropout
        best_valid_pred = best_val_pre
        best_test_pred = best_test_pre

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

        if return_pred == True and return_perf == True:
            return best_valid_perf, best_test_perf, best_valid_pred, best_test_pred, best_iterations
        elif tune_para or return_perf == True:
            return best_valid_perf, best_test_perf, best_iterations
        else:
            return best_valid_pred, best_test_pred, best_iterations
            
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