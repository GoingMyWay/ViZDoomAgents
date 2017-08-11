# coding: utf-8
import pickle

import cv2
import numpy as np
from vizdoom import *


class Recorder(object):
    def __init__(self, episode_num, game, img_shape=(80, 80)):
        if not isinstance(game, ViZDoomGame) or not isinstance(game, DoomGame):
            raise TypeError('TypeError')

        self.episodes = episode_num
        self.game = game
        self.img_shape = img_shape
        self.game.init()
        self.record_buffer = {}

    def play(self):
        for i in range(self.episodes):
            print("Episode #" + str(i + 1))

            buffer = []
            self.game.new_episode()
            st_s = process_frame(self.game.get_state().screen_buffer, self.img_shape)
            s = np.stack((st_s, st_s, st_s, st_s), axis=2)

            while not self.game.is_episode_finished():
                state = self.game.get_state()

                self.game.advance_action()
                last_action = self.game.get_last_action()
                last_reward = self.game.get_last_reward()
                game_variables = state.game_variables

                d = self.game.is_episode_finished()
                if d:
                    s1 = s
                else:
                    img = np.reshape(process_frame(
                        self.game.get_state().screen_buffer, self.img_shape), (*self.img_shape, 1))
                    s1 = np.append(img, s[:, :, :3], axis=2)

                buffer.append([s, last_action, last_reward, s1, d, *game_variables])

            self.record_buffer[i] = buffer

        with open('record.pickle', 'wb') as f:
            pickle.dump(self.record_buffer, f)
            print('Successfully picklized all the buffer data')


class ViZDoomGame(DoomGame):
    def __init__(self, scenario_path):
        super(ViZDoomGame, self).__init__()
        self.load_config(scenario_path)
        self.add_game_args("+freelook 1")
        self.set_screen_resolution(ScreenResolution.RES_640X480)
        self.set_screen_format(ScreenFormat.RGB24)
        self.set_window_visible(True)
        self.set_mode(Mode.SPECTATOR)


def process_frame(frame, img_shape):
    img = cv2.resize(frame, img_shape, interpolation=cv2.INTER_LINEAR)
    img = rgb2gray(img, img_shape)
    return img


def rgb2gray(rgb, img_shape):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray.reshape(*img_shape)


if __name__ == '__main__':
    game = ViZDoomGame(scenario_path='../../scenarios/D3_battle.cfg')
    recorder = Recorder(episode_num=5, game=game)
    try:
        recorder.play()
    except Exception as e:
        print(e)
        print('Error occurred, exit')
    finally:
        game.close()
