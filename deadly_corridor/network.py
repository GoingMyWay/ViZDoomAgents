# coding: utf-8
# implement neural network here

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import configs as cfg

DIM = 128
LSTM_CELL_NUM = 200


class ACNetwork(object):
    """
    Actor-Critic network
    """
    def __init__(self, scope, optimizer, play=False, shape=(80, 80)):
        if not isinstance(optimizer, tf.train.Optimizer) and optimizer is not None:
            raise TypeError("Type Error")
        self.__create_network(scope, optimizer, play=play, shape=shape)

    def __create_network(self, scope, optimizer, play=False, shape=(80, 80)):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, *shape, 1], dtype=tf.float32)
            self.conv_1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.inputs, num_outputs=32,
                                      kernel_size=[8, 8], stride=4, padding='SAME')
            self.conv_2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_1, num_outputs=64,
                                      kernel_size=[4, 4], stride=2, padding='SAME')
            self.conv_3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_2, num_outputs=64,
                                      kernel_size=[3, 3], stride=1, padding='SAME')
            self.fc = slim.fully_connected(slim.flatten(self.conv_3), 512, activation_fn=tf.nn.elu)

            # LSTM
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(cfg.RNN_DIM, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(self.fc, [0])
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                         rnn_in,
                                                         initial_state=state_in,
                                                         sequence_length=step_size,
                                                         time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,
                                               cfg.ACTION_DIM,
                                               activation_fn=tf.nn.softmax,
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,
                                              1,
                                              activation_fn=None,
                                              biases_initializer=None)
            if scope != 'global' and not play:
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, cfg.ACTION_DIM, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, axis=1)

                # Loss functions
                self.policy_loss = -tf.reduce_sum(self.advantages * tf.log(self.responsible_outputs+1e-10))
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy+1e-10))
                self.loss = self.policy_loss + 0.5 * self.value_loss - 0.005 * self.entropy

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                value_var, policy_var = local_vars[:-2] + [local_vars[-1]], local_vars[:-2] + [local_vars[-2]]
                self.var_norms = tf.global_norm(local_vars)

                self.value_gradients = tf.gradients(self.value_loss, value_var)
                value_grads, self.grad_norms_value = tf.clip_by_global_norm(self.value_gradients, 60)

                self.policy_gradients = tf.gradients(self.policy_loss, policy_var)
                policy_grads, self.grad_norms_policy = tf.clip_by_global_norm(self.policy_gradients, 60.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                global_vars_value, global_vars_policy = \
                    global_vars[:-2] + [global_vars[-1]], global_vars[:-2] + [global_vars[-2]]

                self.apply_grads_value = optimizer.apply_gradients(zip(value_grads, global_vars_value))
                self.apply_grads_policy = optimizer.apply_gradients(zip(policy_grads, global_vars_policy))

                # add summary
