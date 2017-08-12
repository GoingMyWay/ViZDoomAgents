# coding: utf-8

IMG_SHAPE = (80, 80)
a_size = 3
model_path = './check_point/supreme/'
model_file = 'model-3750.ckpt'

SCENARIO_PATH = '../scenarios/D3_battle.cfg'

IS_SUPREME_VERSION = True
IS_TRAIN = True
AGENTS_NUM = 16

HIST_LEN = 4


def button_combinations():
    actions = []
    m_forward = [[True], [False]]  # move forward
    m_right_left = [[True, False], [False, True], [False, False]]  # move right and move left
    t_right_left = [[True, False], [False, True], [False, False]]  # turn right and turn left
    attack = [[True], [False]]
    speed = [[True], [False]]

    for i in m_forward:
        for j in m_right_left:
            for k in t_right_left:
                for m in attack:
                    for n in speed:
                        actions.append(i+j+k+m+n)
    return actions

ACTION_DIM = len(button_combinations())
SKIP_FRAME_NUM = 4
