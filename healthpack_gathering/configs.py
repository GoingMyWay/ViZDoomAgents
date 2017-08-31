# coding: utf-8

IMG_SHAPE = (80, 80)
a_size = 3
ACTION_DIM = 2 ** a_size - 2  # remove (True, True, True) and (True, True, False)[Left, Right, Forward]
model_path = './check_point/supreme/'
model_file = 'model-10000.ckpt'

IS_SUPREME_VERSION = True
IS_TRAIN = False
AGENTS_NUM = 16

HIST_LEN = 4
