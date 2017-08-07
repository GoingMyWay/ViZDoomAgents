# coding: utf-8
# implement neural network here
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from . import configs as cfg

DIM = 128
LSTM_CELL_NUM = 200


class ACNetwork(object):
    """
    Actor-Critic network
    """
    def __init__(self, scope, optimizer, play=False):
        if not isinstance(optimizer, tf.train.Optimizer) and optimizer is not None:
            raise TypeError("Type Error")
        self.__create_network(scope, optimizer, play=play)

    def __create_network(self, scope, optimizer, play=False):
        with tf.variable_scope(scope):
            with tf.variable_scope('input_data'):
                self.inputs = tf.placeholder(shape=[None, 80, 80, 1], dtype=tf.float32)
            with tf.variable_scope('networks'):
                with tf.variable_scope('conv_1'):
                    self.conv_1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.inputs, num_outputs=32,
                                              kernel_size=[8, 8], stride=4, padding='SAME')
                with tf.variable_scope('conv_2'):
                    self.conv_2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_1, num_outputs=64,
                                              kernel_size=[4, 4], stride=2, padding='SAME')
                with tf.variable_scope('conv_3'):
                    self.conv_3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_2, num_outputs=64,
                                              kernel_size=[3, 3], stride=1, padding='SAME')
                with tf.variable_scope('f_c'):
                    self.fc = slim.fully_connected(slim.flatten(self.conv_3), 512, activation_fn=tf.nn.elu)

            with tf.variable_scope('actor_critic'):
                with tf.variable_scope('actor'):
                    self.policy = slim.fully_connected(self.fc,
                                                       cfg.ACTION_DIM,
                                                       activation_fn=tf.nn.softmax,
                                                       biases_initializer=None)
                with tf.variable_scope('critic'):
                    self.value = slim.fully_connected(self.fc,
                                                      1,
                                                      activation_fn=None,
                                                      biases_initializer=None)
            if scope != 'global' and not play:
                with tf.variable_scope('action_input'):
                    self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.actions_onehot = tf.one_hot(self.actions, cfg.ACTION_DIM, dtype=tf.float32)
                with tf.variable_scope('target_v'):
                    self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                with tf.variable_scope('advantage'):
                    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, axis=1)
                with tf.variable_scope('loss_func'):
                    with tf.variable_scope('policy_loss'):
                        self.policy_loss = -tf.reduce_sum(self.advantages * tf.log(self.responsible_outputs+1e-10))
                    with tf.variable_scope('value_loss'):
                        self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                    with tf.variable_scope('entropy_loss'):
                        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy+1e-10))
                    with tf.variable_scope('a3c_loss'):
                        self.loss = self.policy_loss + 0.5 * self.value_loss - 0.005 * self.entropy

                with tf.variable_scope('asynchronize'):
                    with tf.variable_scope('get_local_grad'):
                        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                        self.gradients = tf.gradients(self.loss, local_vars)
                        self.var_norms = tf.global_norm(local_vars)
                        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 60.0)

                    with tf.variable_scope('push'):
                        push_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                        self.apply_grads = optimizer.apply_gradients(zip(grads, push_global_vars))

                    with tf.variable_scope('pull'):
                        pull_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                        update_local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                        self.pull_assign_value = [l.assign(g) for g, l in zip(pull_global_vars, update_local_vars)]

    def pull(self, session):
        if not isinstance(session, tf.Session):
            raise TypeError('Invalid Type')

        session.run(self.pull_assign_value)

    def push(self, session, feed_dict):
        if not isinstance(session, tf.Session):
            raise TypeError('Invalid Type')

        session.run(self.apply_grads, feed_dict)

    def get_action_index_and_value(self, session, feed_dict, deterministic=False):
        if not isinstance(session, tf.Session) and not isinstance(feed_dict, dict):
            raise TypeError('Invalid type')

        action_dist, value = session.run([self.policy, self.value], feed_dict)
        a_index = self.choose_action_index(action_dist[0], deterministic=deterministic)
        return a_index, value[0, 0]

    def get_action_index(self, session, feed_dict, deterministic=False):
        if not isinstance(session, tf.Session) and not isinstance(feed_dict, dict):
            raise TypeError('Invalid type')

        action_dist = session.run(self.policy, feed_dict)
        a_index = self.choose_action_index(action_dist[0], deterministic=deterministic)
        return a_index

    def get_value(self, session, feed_dict):
        if not isinstance(session, tf.Session) and not isinstance(feed_dict, dict):
            raise TypeError('Invalid type')

        return session.run(self.value, feed_dict)[0, 0]

    @staticmethod
    def choose_action_index(policy, deterministic=False):
        if deterministic:
            return np.argmax(policy)

        r = random.random()
        cumulative_reward = 0
        for i, p in enumerate(policy):
            cumulative_reward += p
            if r <= cumulative_reward:
                return i

        return len(policy) - 1

    def summarise(self):
        pass
