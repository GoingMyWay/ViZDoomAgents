# coding: utf-8
import importlib

import cv2
import scipy
import pygame
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


class Visualiser(object):
    def __init__(self):
        self.pygame = importlib.import_module('pygame')
        self.screen = None
        self.bg_color = (255, 255, 255)
        self.text_color = (0, 0, 0)

    def init(self):
        self.screen = self.pygame.display.set_mode((680, 480))
        self.pygame.display.set_caption('Battle')

        self.pygame.init()

        self.font_1 = self.pygame.font.SysFont(None, 30)
        self.font_2 = self.pygame.font.SysFont(None, 30)
        self.font_3 = self.pygame.font.SysFont(None, 30)
        self.font_4 = self.pygame.font.SysFont(None, 30)
        self.font_5 = self.pygame.font.SysFont(None, 30)
        self.font_gv = self.pygame.font.SysFont(None, 20)

    def visualize(self, states, game_variables, actions, reward, value):

        self.screen.fill(self.bg_color)

        state_label = self.font_1.render('STATE', 1, self.text_color)
        g_vars_label = self.font_2.render('GAME VARIABLES', 1, self.text_color)
        action_label = self.font_3.render('ACTION', 1, self.text_color)
        reward_label = self.font_4.render('REWARD', 1, self.text_color)
        value_label = self.font_5.render('VALUE', 1, self.text_color)

        self.screen.blit(state_label, (10, 10))
        self.screen.blit(g_vars_label, (10, 130))
        self.screen.blit(action_label, (10, 220))
        self.screen.blit(reward_label, (10, 300))
        self.screen.blit(value_label, (10, 380))

        # states
        pic_1, pic_2, pic_3, pic_4 = \
            map(np.transpose, [states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 3]])
        surf_1 = pygame.surfarray.make_surface(pic_1)
        surf_2 = pygame.surfarray.make_surface(pic_2)
        surf_3 = pygame.surfarray.make_surface(pic_3)
        surf_4 = pygame.surfarray.make_surface(pic_4)

        self.screen.blit(surf_1, (120, 40))
        self.screen.blit(surf_2, (210, 40))
        self.screen.blit(surf_3, (300, 40))
        self.screen.blit(surf_4, (390, 40))

        # game variables
        game_vars_info = self.font_gv.render('AMMO: %s, HEALTH: %s, KILLS: %s' % (
            game_variables[0], game_variables[1], game_variables[2]), 1, self.text_color)
        self.screen.blit(game_vars_info, (170, 170))

        self.pygame.display.flip()
