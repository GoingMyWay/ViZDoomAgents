# coding: utf-8


def button_combinations():
    actions = []
    attack = [[True], [False]]
    m_right_left = [[True, False], [False, True], [False, False]]  # move right and move left
    m_forward = [[True], [False]]  # move forward
    t_right_left = [[True, False], [False, True], [False, False]]  # turn right and turn left
    # s_next_pred_weapon = [[True, False], [False, True], [False, False]]  # select next and previous weapon

    for i in attack:
        for j in m_right_left:
            for k in m_forward:
                for m in t_right_left:
                    actions.append(i+j+k+m)
    return actions


img_shape = (80, 80)
a_size = 7
RNN_DIM = 256
new_img_shape = (120, 120)
BUFFER_SIZE = 64
ACTION_DIM = len(button_combinations())
