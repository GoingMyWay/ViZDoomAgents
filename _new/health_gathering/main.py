#!/usr/bin/env python3
# coding: utf-8
import time
import sys
import string
import threading

import os
import tensorflow as tf

from vizdoom import *

from . import agent
from . import network
from . import configs as cfg


def main_train(tf_configs=None):
    s_t = time.time()

    tf.reset_default_graph()

    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    sess = tf.Session(config=tf_configs)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)
        master_network = network.ACNetwork('global', optimizer)  # Generate global network
        num_workers = 4
        agents = []
        # Create worker classes
        for i in range(num_workers):
            agents.append(agent.Agent(sess, agent.Game(cfg.SCENARIO_PATH, play=False), i, optimizer, global_episodes))
        saver = tf.train.Saver(max_to_keep=100)

    coord = tf.train.Coordinator()
    if cfg.load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(config.MODEL_PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for ag in agents:
        t = threading.Thread(target=lambda: ag.train_a3c(sess, coord, saver))
        t.start()
        time.sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)

    sess.close()

    print("training ends, costs{}".format(time.time() - s_t))


def main_play(tf_configs=None):
    tf.reset_default_graph()

    with tf.Session(config=tf_configs) as sess:

        ag = agent.Agent(sess, agent.Game(cfg.SCENARIO_PATH, play=True), 0, play=True)
        print('Loading Model...')
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)
        saver.restore(sess, os.path.join(cfg.MODEL_PATH, 'model-1750.ckpt'))
        print('Successfully loaded!')

        ag.play_game(sess, 10)


if __name__ == '__main__':

    train = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if train:
        main_train(tf_configs=config)
    else:
        main_play(tf_configs=config)
