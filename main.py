import popgym
from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
from popgym.core.observability import Observability, STATE
import math
import numpy as np
import copy
from tqdm import tqdm
from mcts import run_simulation_mcts
from sparse_sampling import forward_search
import time

actions = [0, 1, 2, 3]

if __name__ == '__main__':
    env_classes = popgym.envs.ALL.keys()
    env_popgym = popgym.envs.labyrinth_escape.LabyrinthEscapeMedium()
    wrapped_env = PreviousAction(env_popgym)
    wrapped_env = Antialias(wrapped_env)
    DiscreteAction(Flatten(wrapped_env))
    wrapped_env = Markovian(wrapped_env, Observability.FULL_IN_INFO_DICT)

    run_count = 50

    # mcts

    table = ("| Number of simulation | Success rate under 15 steps | Average number of steps for successful "
             "trajectory | Seconds per run |")
    for s in []:

        seed = 42
        success = 0
        traj_len = 0
        start_time = time.time()
        for i in tqdm(range(run_count)):
            wrapped_env.reset(seed=seed)
            seed += 1
            obs, _, terminated, truncated, info = wrapped_env.step(1)

            traj, reward = run_simulation_mcts(wrapped_env, info['position'], obs, simulations=s, depth=25, steps=25)

            if reward > 0:
                success += 1
                traj_len += len(traj)

        end_time = time.time()
        avg_runtime = (end_time - start_time) / run_count

        line = f"| {s} | {success / run_count:.2f} | {traj_len / success:.2f} | {avg_runtime:.2f} |\n"
        table += line
        print(line)

    print(table)

    # forward search
    table += "\n\n| Depth | Success rate | Average number of steps for successful trajectory | Seconds per run |"

    for d in [11]:
        seed = 42
        success = 0
        traj_len = 0
        start_time = time.time()
        for i in tqdm(range(run_count)):
            wrapped_env.reset(seed=seed)
            seed += 1
            obs, _, terminated, truncated, info = wrapped_env.step(1)

            traj = forward_search(wrapped_env, info['position'], obs, steps=0, depth=d)

            if traj is not None:
                success += 1
                traj_len += len(traj)

        end_time = time.time()
        avg_runtime = (end_time - start_time) / run_count

        line = f"| {d} | {success / run_count:.2f} | {traj_len / success:.2f} | {avg_runtime:.2f} |\n"
        table += line
        print(line)
    print(table)
