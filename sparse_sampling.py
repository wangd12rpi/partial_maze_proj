import popgym
from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
from popgym.core.observability import Observability, STATE
import math
import numpy as np
import copy
from tqdm import tqdm
from utils import get_valid_actions

actions = [0, 1, 2, 3]

import math
import copy
import random


def forward_search(env, current_state, obs, steps, depth, done=False):
    if steps >= depth:
        return None

    if done:
        return []

    valid_actions = get_valid_actions(obs)

    for action in valid_actions:
        env_copy = copy.deepcopy(env)
        next_obs, reward, terminated, truncated, info = env_copy.step(action)
        next_state = info['position']
        result = forward_search(env_copy, next_state, next_obs, steps + 1, depth, reward > 0)
        if result is not None:
            return [action] + result

    return None
