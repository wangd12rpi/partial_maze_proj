import popgym
from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
from popgym.core.observability import Observability, STATE
import math
import numpy as np
import copy
from tqdm import tqdm


def get_valid_actions(observation_l):
    actions = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }
    observation = observation_l[0][:9].tolist()

    agent_index = 4
    agent_row = agent_index // 3
    agent_col = agent_index % 3

    valid_actions = []

    for action in actions.keys():
        (dx, dy) = actions[action]
        new_row = agent_row + dy
        new_col = agent_col + dx
        new_index = new_row * 3 + new_col

        if new_index < 0 or observation[new_index] == 1:
            pass
        else:
            valid_actions.append(action)

    return valid_actions
