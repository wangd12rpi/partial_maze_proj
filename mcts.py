import popgym
from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
from popgym.core.observability import Observability, STATE
import math
import numpy as np
import copy
from tqdm import tqdm
from utils import get_valid_actions

actions = [0, 1, 2, 3]


# mcts
def mcts(env, state, obs, depth, simulations, gamma, exploration_param=10):
    N = {}
    Q = {}

    def explore(state, exp_obs):
        valid_act = actions
        n_sum = sum(N[(state, a_)] for a_ in valid_act)
        if n_sum == 0:
            n_sum = 100

        top_act = sorted(
            valid_act,
            key=lambda a: Q[(state, a)] + exploration_param * math.sqrt(math.log(n_sum) / (N[(state, a)] + 1e-4)),
            reverse=True
        )
        return top_act[0]

    def simulate(env_cpy, state, obs, r, d):
        if d <= 0 or r > 0:
            return r

        if (state, 0) not in N and (state, 1) not in N and (state, 2) not in N and (
                state, 3) not in N:
            for a in actions:
                N[(state, a)] = 0
                Q[(state, a)] = 0.0
            return r

        act = explore(state, obs)
        obser, r, terminated, truncated, info = env_cpy.step(act)
        next_state = info['position']
        q = r + gamma * simulate(env_cpy, next_state, obser, r, d - 1)
        N[(state, act)] += 1
        Q[(state, act)] += (q - Q[(state, act)]) / N[(state, act)]
        return q

    for _ in range(simulations):
        env_cpy = copy.deepcopy(env)
        simulate(env_cpy, state, obs, 0, depth)

    ranked_act = sorted(actions, key=lambda a: Q[(state, a)], reverse=True)
    return ranked_act[0]


def simulate_mcts_step(env, current_state, obs, depth, simulations, gamma, exploration_param):
    best_action = mcts(env, current_state, obs, depth, simulations, gamma)

    obs, reward, terminated, truncated, info = env.step(best_action)

    if reward > 0:
        return current_state, None, reward, obs

    next_state = info['position']
    return next_state, best_action, reward, obs


def run_simulation_mcts(env, current_state, obs, steps, depth, simulations, gamma=0.95, exploration_param=10):
    trajectory = []

    final_reward = -10
    for _ in range(steps):

        next_state, action, r, next_obs = simulate_mcts_step(env, current_state, obs, depth=depth, gamma=gamma,
                                                             simulations=simulations,
                                                             exploration_param=exploration_param)
        final_reward = r
        trajectory.append((current_state, action))

        if next_state is None or action is None:
            break

        current_state = next_state
        obs = next_obs

    return trajectory, final_reward
