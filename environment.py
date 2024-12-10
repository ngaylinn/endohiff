"""The Environment data type and definitions for all control environments.
"""

import numpy as np
import taichi as ti

from constants import (
    BITSTR_POWER, BITSTR_LEN,  ENVIRONMENT_SHAPE, NUM_WEIGHTS, MIN_HIFF,
    MAX_HIFF)


@ti.data_oriented
class Environment:
    def __init__(self):
        # The threshold for surviving at each location in the environment.
        self.min_fitness = ti.field(
            dtype=float, shape=ENVIRONMENT_SHAPE)

        # The substring weights to pass into the weighted_hiff function, for
        # each location in the environment.
        self.weights = ti.Vector.field(
            n=NUM_WEIGHTS, dtype=float, shape=ENVIRONMENT_SHAPE)

    def to_numpy(self):
        return {
            'min_fitness': self.min_fitness.to_numpy(),
            'weights': self.weights.to_numpy()
        }


def make_flat():
    env = Environment()
    env.min_fitness.fill(0.0)
    env.weights.fill(1.0)
    return env


def make_random():
    env = Environment()
    env.min_fitness.from_numpy(
        np.random.rand(*ENVIRONMENT_SHAPE).astype(np.float32) * MAX_HIFF)
    env.weights.from_numpy(
        np.random.rand(*ENVIRONMENT_SHAPE, NUM_WEIGHTS).astype(np.float32))
    return env


def make_baym():
    ew, _ = ENVIRONMENT_SHAPE
    buckets = 2 * BITSTR_POWER - 1
    bucket_width = ew // buckets
    ramp_up = np.arange(MIN_HIFF, MAX_HIFF, BITSTR_LEN).repeat(bucket_width)
    ramp_down = np.flip(ramp_up)
    ramp_up_and_down = np.full(ew, MAX_HIFF)
    ramp_up_and_down[:ramp_up.size] = ramp_up
    ramp_up_and_down[-ramp_down.size:] = ramp_down

    env = Environment()
    env.min_fitness.from_numpy(
        np.broadcast_to(
            np.expand_dims(ramp_up_and_down, 1),
            ENVIRONMENT_SHAPE))
    env.weights.fill(1.0)
    return env


# The named environments to experiment with.
ENVIRONMENTS = {
    'flat': make_flat,
    'random': make_random,
    'baym': make_baym,
}


# A demo to visualize any of the environments defined above.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from render import render_env_map_min_fitness

    ti.init(ti.cuda, unrolling_limit=0)

    env = make_baym()
    render_env_map_min_fitness('baym', env.to_numpy())
    plt.show()
