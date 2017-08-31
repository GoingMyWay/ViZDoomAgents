# coding: utf-8
import cv2
import scipy
import numpy as np
import tensorflow as tf
import scipy.signal as signal


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame, img_shape):
    img = cv2.resize(frame, img_shape, interpolation=cv2.INTER_LINEAR)
    img = rgb2gray(img, img_shape)
    return img


def rgb2gray(rgb, img_shape):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray.reshape(*img_shape)
