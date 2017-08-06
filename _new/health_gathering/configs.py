# coding: utf-8


s_size = 6400  # 80 * 80 * 1

img_dim = 80
a_size = 3  # LEFT, RIGHT, FORWARD
gamma = .99
load_model = False
max_episode_length = 2100
MODEL_PATH = 'check_point'
SCENARIO_PATH = '../scenarios/health_gathering.wad'
ACTION_DIM = 2 ** a_size - 2  # remove (True, True, True) and (True, True, False)[Left, Right, Forward]
