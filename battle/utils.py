# coding: utf-8
import importlib

import cv2
import scipy
import pygame
import numpy as np
import tensorflow as tf
import scipy.signal as signal

import configs as cfg


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
        self.head_color = (148, 0, 211)

    def init(self):
        self.screen = self.pygame.display.set_mode((900, 800))
        self.pygame.display.set_caption('Battle')

        self.pygame.init()

        self.font_1 = self.pygame.font.SysFont(None, 30)
        self.font_2 = self.pygame.font.SysFont(None, 30)
        self.font_3 = self.pygame.font.SysFont(None, 30)
        self.font_4 = self.pygame.font.SysFont(None, 30)
        self.font_5 = self.pygame.font.SysFont(None, 30)
        self.font_gv = self.pygame.font.SysFont(None, 25)
        self.font_reward = self.pygame.font.SysFont(None, 25)
        self.font_value = self.pygame.font.SysFont(None, 25)
        self.font_scalar = self.pygame.font.SysFont(None, 15)

        self.forward = pygame.image.load(cfg.forward)
        self.forward_back = pygame.image.load(cfg.forward_back)
        self.left = pygame.image.load(cfg.left)
        self.left_back = pygame.image.load(cfg.left_back)
        self.right = pygame.image.load(cfg.right)
        self.right_back = pygame.image.load(cfg.right_back)
        self.turn_left = pygame.image.load(cfg.turn_left)
        self.turn_left_back = pygame.image.load(cfg.turn_left_back)
        self.turn_right = pygame.image.load(cfg.turn_right)
        self.turn_right_back = pygame.image.load(cfg.turn_right_back)
        self.attack = pygame.image.load(cfg.attack)
        self.attack_back = pygame.image.load(cfg.attack_back)
        self.speed = pygame.image.load(cfg.speed)
        self.speed_back = pygame.image.load(cfg.speed_back)

    def visualize(self, states, game_variables, actions, reward_list, value_list):

        self.screen.fill(self.bg_color)

        state_label = self.font_1.render('STATE', 1, self.head_color)
        g_vars_label = self.font_2.render('GAME VARIABLES', 1, self.head_color)
        action_label = self.font_3.render('ACTION', 1, self.head_color)
        reward_label = self.font_4.render('REWARD', 1, self.head_color)
        value_label = self.font_5.render('VALUE', 1, self.head_color)

        self.screen.blit(state_label, (10, 10))
        self.screen.blit(g_vars_label, (10, 130))
        self.screen.blit(action_label, (10, 220))
        self.screen.blit(reward_label, (10, 320))
        self.screen.blit(value_label, (10, 520))

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

        # actions
        if actions[0]:
            self.screen.blit(self.forward, (140, 250))
        else:
            self.screen.blit(self.forward_back, (140, 250))

        if actions[1]:
            self.screen.blit(self.left, (60, 250))
        else:
            self.screen.blit(self.left_back, (60, 250))

        if actions[2]:
            self.screen.blit(self.right, (220, 250))
        else:
            self.screen.blit(self.right_back, (220, 250))

        if actions[3]:
            self.screen.blit(self.turn_left, (300, 250))
        else:
            self.screen.blit(self.turn_left_back, (300, 250))

        if actions[4]:
            self.screen.blit(self.turn_right, (380, 250))
        else:
            self.screen.blit(self.turn_right_back, (380, 250))

        if actions[5]:
            self.screen.blit(self.attack, (460, 250))
        else:
            self.screen.blit(self.attack_back, (460, 250))

        if actions[6]:
            self.screen.blit(self.speed, (540, 250))
        else:
            self.screen.blit(self.speed_back, (540, 250))

        # dynamic rewards
        reward_info = self.font_reward.render('Reward: %s' % reward_list[-1], 1, self.text_color)
        self.screen.blit(reward_info, (170, 505))
        self._reward_line_char(reward_list)

        # dynamic values
        value_info = self.font_value.render('Value: %s' % value_list[-1], 1, self.text_color)
        self.screen.blit(value_info, (170, 730))
        self._value_line_char(value_list)

        self.pygame.display.flip()

    def _reward_line_char(self, rewards):
        self.pygame.draw.line(self.screen, (0, 0, 0), (50, 350), (50, 500), 1)  # top left -> bottom left
        self.pygame.draw.line(self.screen, (0, 0, 0), (890, 350), (890, 500), 1)  # top right -> bottom right
        self.pygame.draw.line(self.screen, (0, 0, 0), (50, 350), (890, 350), 1)  # top left -> top right
        self.pygame.draw.line(self.screen, (0, 0, 0), (50, 500), (890, 500), 1)  # bottom left -> bottom right
        max_v, min_v = max(rewards), min(rewards)
        mid_v = (max_v + min_v) / 2.
        max_v_infor = self.font_scalar.render('max: %s' % max_v, 1, self.text_color)
        min_v_infor = self.font_scalar.render('min: %s' % min_v, 1, self.text_color)
        mid_v_infor = self.font_scalar.render('mid: %s' % mid_v, 1, self.text_color)
        self.screen.blit(max_v_infor, (5, 350))
        self.screen.blit(min_v_infor, (5, 500))
        self.screen.blit(mid_v_infor, (5, 425))

        # x axis and y axis
        x_cords = [50+0.4*v for v in range(cfg.MAX_TIME_OUT_STEP)]
        if max_v - min_v != 0:
            y_cords = [350+(max_v-v)/(max_v-min_v) * (500-350) for v in rewards]
        else:
            y_cords = [350+(500-350)/2. for _ in rewards]

        # draw lines
        for _i in range(len(rewards)-1):
            self.pygame.draw.line(
                self.screen, (0, 0, 255), (x_cords[_i], y_cords[_i]), (x_cords[_i+1], y_cords[_i+1]), 2)

        # draw step number
        _color = (0, 0, 255)
        step_info = self.font_scalar.render('%s' % str(len(rewards)+cfg.SKIP_FRAME_NUM-1), 1, _color)
        self.screen.blit(step_info, (x_cords[len(rewards)], 503))

    def _value_line_char(self, values):
        self.pygame.draw.line(self.screen, (0, 0, 0), (50, 550), (50, 700), 1)  # top left -> bottom left
        self.pygame.draw.line(self.screen, (0, 0, 0), (890, 550), (890, 700), 1)  # top right -> bottom right
        self.pygame.draw.line(self.screen, (0, 0, 0), (50, 550), (890, 550), 1)  # top left -> top right
        self.pygame.draw.line(self.screen, (0, 0, 0), (50, 700), (890, 700), 1)  # bottom left -> bottom right
        max_v, min_v = max(values), min(values)
        mid_v = (max_v + min_v) / 2.
        max_v_infor = self.font_scalar.render('max: %s' % max_v, 1, self.text_color)
        min_v_infor = self.font_scalar.render('min: %s' % min_v, 1, self.text_color)
        mid_v_infor = self.font_scalar.render('mid: %s' % mid_v, 1, self.text_color)
        self.screen.blit(max_v_infor, (5, 550))
        self.screen.blit(min_v_infor, (5, 700))
        self.screen.blit(mid_v_infor, (5, 625))

        # x axis and y axis
        x_cords = [50+0.4*v for v in range(cfg.MAX_TIME_OUT_STEP)]
        if max_v - min_v != 0:
            y_cords = [550+(max_v-v)/(max_v-min_v)*(700-550) for v in values]
        else:
            y_cords = [550+(700-550)/2. for _ in values]

        # draw lines
        for _i in range(len(values)-1):
            self.pygame.draw.line(
                self.screen, (255, 0, 0), (x_cords[_i], y_cords[_i]), (x_cords[_i+1], y_cords[_i+1]), 2)

        # draw step number
        _color = (255, 0, 0)
        step_info = self.font_scalar.render('%s' % str(len(values)+cfg.SKIP_FRAME_NUM-1), 1, _color)
        self.screen.blit(step_info, (x_cords[len(values)], 703))
