#Copyright 2018 Daniel Selsam. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import numpy as np
import math
import random
import os
import time
from confusion import ConfusionMatrix
from problems_loader import init_problems_loader
from mlp import MLP
from util import repeat_end, decode_final_reducer, decode_transfer_fn
from tensorflow.contrib.rnn import LSTMStateTuple
# from sklearn.cluster import KMeans

class NeuroSAT(object):
    def __init__(self, opts):
        self.opts = opts

        self.final_reducer = decode_final_reducer(opts.final_reducer)

        self.build_network()
        self.train_problems_loader = None

    def init_random_seeds(self):
        tf.set_random_seed(self.opts.tf_seed)
        np.random.seed(self.opts.np_seed)

    def construct_session(self):
        self.sess = tf.Session()

    def declare_parameters(self):
        opts = self.opts
        with tf.variable_scope('params') as scope:
            self.A_init = tf.get_variable(name="A_init", initializer=tf.random_normal([1, self.opts.d]))
            self.L_init = tf.get_variable(name="L_init", initializer=tf.random_normal([1, self.opts.d]))
            self.CA_init = tf.get_variable(name="CA_init", initializer=tf.random_normal([1, self.opts.d]))
            self.CL_init = tf.get_variable(name="CL_init", initializer=tf.random_normal([1, self.opts.d]))

            self.A_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("A_msg"))
            self.L_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("L_msg"))
            self.CA_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("CA_msg"))
            self.CL_msg = MLP(opts, opts.d, repeat_end(opts.d, opts.n_msg_layers, opts.d), name=("CL_msg"))

            self.A_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.L_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.CA_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))
            self.CL_update = tf.contrib.rnn.LayerNormBasicLSTMCell(self.opts.d, activation=decode_transfer_fn(opts.lstm_transfer_fn))

            self.A_vote = MLP(opts, opts.d, repeat_end(opts.d, opts.n_vote_layers, 1), name=("A_vote"))
            # self.L_vote = MLP(opts, opts.d, repeat_end(opts.d, opts.n_vote_layers, 1), name=("L_vote"))
            # self.vote_bias = tf.get_variable(name="vote_bias", shape=[], initializer=tf.zeros_initializer())

    def declare_placeholders(self):
        self.n_A_vars = tf.placeholder(tf.int32, shape=[], name='n_A_vars')
        self.n_A_lits = tf.placeholder(tf.int32, shape=[], name='n_A_lits')
        self.n_L_vars = tf.placeholder(tf.int32, shape=[], name='n_L_vars')
        self.n_L_lits = tf.placeholder(tf.int32, shape=[], name='n_L_lits')
        self.n_clauses = tf.placeholder(tf.int32, shape=[], name='n_clauses')

        self.A_unpack = tf.sparse_placeholder(tf.float32, shape=[None, None], name='A_unpack')
        self.L_unpack = tf.sparse_placeholder(tf.float32, shape=[None, None], name='L_unpack')
        # self.is_sat = tf.placeholder(tf.bool, shape=[None], name='is_sat')
        self.labels = tf.placeholder(tf.int32, shape=[None, None, 2], name='labels')

        # useful helpers
        # self.n_batches = tf.shape(self.is_sat)[0]
        self.n_batches = tf.shape(self.labels)[0]
        self.n_A_vars_per_batch = tf.div(self.n_A_vars, self.n_batches)
        self.n_L_vars_per_batch = tf.div(self.n_L_vars, self.n_batches)
        # self.n_A_vars_per_batch = tf.placeholder(tf.int32, shape=[], name='n_A_vars_per_batch')
        # self.n_L_vars_per_batch = tf.placeholder(tf.int32, shape=[], name='n_L_vars_per_batch')

    def while_cond(self, i, A_state, L_state, CA_state, CL_state):
        return tf.less(i, self.opts.n_rounds)

    def flip(self, lits, size):
        return tf.concat([lits[size : (2 * size), :], lits[0 : size, :]], axis=0)

    def while_body(self, i, A_state, L_state, CA_state, CL_state):
        A_pre_msgs = self.A_msg.forward(A_state.h)
        AC_msgs = tf.sparse_tensor_dense_matmul(self.A_unpack, A_pre_msgs, adjoint_a=True)

        L_pre_msgs = self.L_msg.forward(L_state.h)
        LC_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, L_pre_msgs, adjoint_a=True)

        ALC_msgs = tf.concat([AC_msgs, LC_msgs], axis=1)
        with tf.variable_scope('CA_update') as scope:
            _, CA_state = self.CA_update(inputs=ALC_msgs, state=CA_state)
        with tf.variable_scope('CL_update') as scope:
            _, CL_state = self.CL_update(inputs=ALC_msgs, state=CL_state)

        CA_pre_msgs = self.CA_msg.forward(CA_state.h)
        CL_pre_msgs = self.CL_msg.forward(CL_state.h)
        CA_msgs = tf.sparse_tensor_dense_matmul(self.A_unpack, CA_pre_msgs)
        CL_msgs = tf.sparse_tensor_dense_matmul(self.L_unpack, CL_pre_msgs)

        with tf.variable_scope('A_update') as scope:
            _, A_state = self.A_update(inputs=tf.concat([CA_msgs, self.flip(A_state.h, self.n_A_vars)], axis=1), state=A_state)
        with tf.variable_scope('L_update') as scope:
            _, L_state = self.L_update(inputs=tf.concat([CL_msgs, self.flip(L_state.h, self.n_L_vars)], axis=1), state=L_state)

        return i+1, A_state, L_state, CA_state, CL_state

    def pass_messages(self):
        with tf.name_scope('pass_messages') as scope:
            denom = tf.sqrt(tf.cast(self.opts.d, tf.float32))

            A_output = tf.tile(tf.div(self.A_init, denom), [self.n_A_lits, 1])
            L_output = tf.tile(tf.div(self.L_init, denom), [self.n_L_lits, 1])
            CA_output = tf.tile(tf.div(self.CA_init, denom), [self.n_clauses, 1])
            CL_output = tf.tile(tf.div(self.CL_init, denom), [self.n_clauses, 1])

            A_state = LSTMStateTuple(h=A_output, c=tf.zeros([self.n_A_lits, self.opts.d]))
            L_state = LSTMStateTuple(h=L_output, c=tf.zeros([self.n_L_lits, self.opts.d]))
            CA_state = LSTMStateTuple(h=CA_output, c=tf.zeros([self.n_clauses, self.opts.d]))
            CL_state = LSTMStateTuple(h=CL_output, c=tf.zeros([self.n_clauses, self.opts.d]))

            _, A_state, L_state, CA_state, CL_state = tf.while_loop(self.while_cond, self.while_body, [0, A_state, L_state, CA_state, CL_state])

        self.final_A_lits = A_state.h
        # self.final_L_lits = L_state.h
        # self.final_clauses = C_state.h
        
    def compute_logits(self):
        with tf.name_scope('compute_logits') as scope:
            self.all_votes_A = self.A_vote.forward(self.final_A_lits) # n_A_lits x 1
            # self.all_votes_L = self.L_vote.forward(self.final_L_lits) # n_L_lits x 1
            self.all_votes_join_A = tf.concat([self.all_votes_A[0:self.n_A_vars], self.all_votes_A[self.n_A_vars:self.n_A_lits]], axis=1)
            # self.all_votes_join_L = tf.concat([self.all_votes_L[0:self.n_L_vars], self.all_votes_L[self.n_L_vars:self.n_L_lits]], axis=1)
            self.all_votes_batched_A = tf.reshape(self.all_votes_join_A, [self.n_batches, self.n_A_vars_per_batch, 2])
            # self.all_votes_batched_L = tf.reshape(self.all_votes_join_L, [self.n_batches, self.n_L_vars_per_batch, 2])

            # try to use only A_votes for logits?
            self.logits = self.all_votes_batched_A 
            # self.final_reducer(self.all_votes_batched_A) + self.vote_bias# + self.final_reducer(self.all_votes_batched_L) 

    def compute_cost(self):
        # self.predict_costs = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.labels, tf.float32))
        self.predict_costs = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=tf.cast(self.labels, tf.float32))
        self.predict_cost = tf.reduce_mean(self.predict_costs)

        with tf.name_scope('l2') as scope:
            l2_cost = tf.zeros([])
            for var in tf.trainable_variables():
                l2_cost += tf.nn.l2_loss(var)

        self.cost = tf.identity(self.predict_cost + self.opts.l2_weight * l2_cost, name="cost")

    def build_optimizer(self):
        opts = self.opts

        self.global_step = tf.get_variable("global_step", shape=[], initializer=tf.zeros_initializer(), trainable=False)

        if opts.lr_decay_type == "no_decay":
            self.learning_rate = tf.constant(opts.lr_start)
        elif opts.lr_decay_type == "poly":
            self.learning_rate = tf.train.polynomial_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_end, power=opts.lr_power)
        elif opts.lr_decay_type == "exp":
            self.learning_rate = tf.train.exponential_decay(opts.lr_start, self.global_step, opts.lr_decay_steps, opts.lr_decay, staircase=False)
        else:
            raise Exception("lr_decay_type must be 'no_decay', 'poly' or 'exp'")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.opts.clip_val)
        self.apply_gradients = optimizer.apply_gradients(zip(gradients, variables), name='apply_gradients', global_step=self.global_step)

    def initialize_vars(self):
        tf.global_variables_initializer().run(session=self.sess)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.opts.n_saves_to_keep)
        if self.opts.run_id is not None:
            self.save_dir = "snapshots/run%d" % self.opts.run_id
            self.save_prefix = os.path.join(self.save_dir, "snap") #"%s/snap" % self.save_dir

    def build_network(self):
        self.init_random_seeds()
        self.construct_session()
        self.declare_parameters()
        self.declare_placeholders()
        self.pass_messages()
        self.compute_logits()
        self.compute_cost()
        self.build_optimizer()
        self.initialize_vars()
        self.init_saver()
        if self.opts.restore_id is not None:
            if self.opts.restore_epoch is not None:
                self.restore()

    def save(self, epoch):
        self.saver.save(self.sess, self.save_prefix, global_step=epoch)

    def restore(self):
        snapshot = "snapshots/run%d/snap-%d" % (self.opts.restore_id, self.opts.restore_epoch)
        print('restoring from {}'.format(snapshot))
        self.saver.restore(self.sess, snapshot)

    def build_feed_dict(self, problem):
        d = {}
        d[self.n_L_vars] = problem.n_vars_AL[1]
        d[self.n_L_lits] = problem.n_lits_AL[1]
        d[self.n_A_vars] = problem.n_vars_AL[0]
        d[self.n_A_lits] = problem.n_lits_AL[0]
        d[self.n_clauses] = problem.n_clauses

        d[self.L_unpack] =  tf.SparseTensorValue(indices=problem.L_unpack_indices,
                                                 values=np.ones(problem.L_unpack_indices.shape[0]),
                                                 dense_shape=[problem.n_lits_AL[1], problem.n_clauses])
        d[self.A_unpack] =  tf.SparseTensorValue(indices=problem.A_unpack_indices,
                                                 values=np.ones(problem.A_unpack_indices.shape[0]),
                                                 dense_shape=[problem.n_lits_AL[0], problem.n_clauses])

        # d[self.is_sat] = problem.is_sat
        d[self.labels] = problem.labels
        return d

    def train_epoch(self, epoch):
        if self.train_problems_loader is None:
            self.train_problems_loader = init_problems_loader(self.opts.train_dir)

        epoch_start = time.clock()

        epoch_train_cost = 0.0
        accuracy_by_var = []
        accuracy_by_problem = []
        # epoch_train_mat = ConfusionMatrix()

        train_problems, train_filename = self.train_problems_loader.get_next()
        for problem in train_problems:
            d = self.build_feed_dict(problem)
            _, logits, cost = self.sess.run([self.apply_gradients, self.logits, self.cost], feed_dict=d)
            epoch_train_cost += cost
            (av, ap) = self.accuracy(np.array(logits), np.array(problem.labels))
            accuracy_by_var.append(av)
            accuracy_by_problem.append(ap)
            # epoch_train_mat.update(problem.is_sat, logits > 0)

        epoch_train_cost /= len(train_problems)
        av_ac_by_var = np.mean(accuracy_by_var)
        av_ac_by_pro = np.mean(accuracy_by_problem)
        # epoch_train_mat = epoch_train_mat.get_percentages()
        epoch_end = time.clock()

        learning_rate = self.sess.run(self.learning_rate)
        self.save(epoch)

        return (train_filename, epoch_train_cost, av_ac_by_var, av_ac_by_pro, learning_rate, epoch_end - epoch_start)

    def accuracy(self, logits, labels):
        assert(logits.shape == labels.shape), 'shape of logits and labels are not the same {} != {}'.format(logits.shape, labels.shape)
        logit_assign = np.argmax(logits, axis = -1)
        label_assign = np.argmax(labels, axis = -1)
        same = np.equal(logit_assign, label_assign)
        accuracy_by_var = np.sum(same.astype(np.float32)) / np.sum(np.ones_like(same, dtype=np.float32))
        all_same = np.all(same, axis = -1)
        accuracy_by_problem = np.sum(all_same.astype(np.float32)) / np.sum(np.ones_like(all_same, dtype=np.float32))
        return accuracy_by_var, accuracy_by_problem

    def test(self, test_data_dir):
        test_problems_loader = init_problems_loader(test_data_dir)
        results = []

        while test_problems_loader.has_next():
            test_problems, test_filename = test_problems_loader.get_next()

            epoch_test_cost = 0.0
            accuracy_by_vars = []
            accuracy_by_prob = []
            # epoch_test_mat = ConfusionMatrix()

            for problem in test_problems:
                d = self.build_feed_dict(problem)
                logits, cost = self.sess.run([self.logits, self.cost], feed_dict=d)
                epoch_test_cost += cost
                (ac_by_var, ac_by_pro) = self.accuracy(np.array(logits), np.array(problem.labels))
                accuracy_by_vars.append(ac_by_var)
                accuracy_by_prob.append(ac_by_pro)
                # epoch_test_mat.update(problem.is_sat, logits > 0)

            epoch_test_cost /= len(test_problems)
            av_ac_by_var = np.mean(accuracy_by_vars)
            av_ac_by_pro = np.mean(accuracy_by_prob)
            # epoch_test_mat = epoch_test_mat.get_percentages()

            results.append((test_filename, epoch_test_cost, av_ac_by_var, av_ac_by_pro))

        return results

    def find_solutions(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars

        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        d = self.build_feed_dict(problem)
        all_votes, final_lits, logits, costs = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

        solutions = []
        for batch in range(len(problem.is_sat)):
            decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
            decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))

            def reify(phi):
                xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                              [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                def one_of(a, b): return (a and (not b)) or (b and (not a))
                assert(all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]

            if self.solves(problem, batch, decode_cheap_A): solutions.append(reify(decode_cheap_A))
            elif self.solves(problem, batch, decode_cheap_B): solutions.append(reify(decode_cheap_B))
            else:

                L = np.reshape(final_lits, [2 * n_batches, n_vars_per_batch, self.opts.d])
                L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

                kmeans = KMeans(n_clusters=2, random_state=0).fit(L)
                distances = kmeans.transform(L)
                scores = distances * distances

                def proj_vlit_flit(vlit):
                    if vlit < problem.n_vars: return vlit - batch * n_vars_per_batch
                    else:                     return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

                def decode_kmeans_A(vlit):
                    return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                        scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]

                decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

                if self.solves(problem, batch, decode_kmeans_A): solutions.append(reify(decode_kmeans_A))
                elif self.solves(problem, batch, decode_kmeans_B): solutions.append(reify(decode_kmeans_B))
                else: solutions.append(None)

        return solutions

    def solves(self, problem, batch, phi):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]

        if start_cell == end_cell:
            # no clauses
            return 1.0

        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False

        for cell in range(start_cell, end_cell):
            next_clause = problem.L_unpack_indices[cell, 1]

            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    return False

                current_clause = next_clause
                current_clause_satisfied = False

            if not current_clause_satisfied:
                vlit = problem.L_unpack_indices[cell, 0]
                #print("[%d] %d" % (batch, vlit))
                if phi(vlit):
                    current_clause_satisfied = True

        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied: return False
        return True
