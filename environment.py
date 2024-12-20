"""The Environments data type and definitions for all control environments.
"""

import numpy as np
import taichi as ti

from constants import (
    BITSTR_POWER, BITSTR_LEN,  ENVIRONMENT_SHAPE, NUM_WEIGHTS, MIN_HIFF,
    MAX_HIFF)


@ti.data_oriented
class Environments:
    def __init__(self, count=1):
        self.shape = (count,) + ENVIRONMENT_SHAPE

        # The threshold for surviving at each location in the environment.
        self.min_fitness = ti.field(dtype=float, shape=self.shape)

        # The substring weights to pass into the weighted_hiff function, for
        # each location in the environment.
        self.weights = ti.Vector.field(
            n=NUM_WEIGHTS, dtype=float, shape=self.shape)

    def __getitem__(self, i):
        return {
            'min_fitness': self.min_fitness.to_numpy()[i],
            'weights': self.weights.to_numpy()[i]
        }


def make_flat(count=1):
    env = Environments(count)
    env.min_fitness.fill(0.0)
    env.weights.fill(1.0)
    return env


def make_random(count=1):
    env = Environments(count)
    env.min_fitness.from_numpy(
        np.random.rand(*env.shape).astype(np.float32) * MAX_HIFF)
    env.weights.from_numpy(
        np.random.rand(*env.shape, NUM_WEIGHTS).astype(np.float32))
    return env


def make_baym(count=1):
    ew, _ = ENVIRONMENT_SHAPE
    buckets = 2 * BITSTR_POWER - 1
    bucket_width = ew // buckets
    ramp_up = np.arange(MIN_HIFF, MAX_HIFF, BITSTR_LEN).repeat(bucket_width)
    ramp_down = np.flip(ramp_up)
    ramp_up_and_down = np.full(ew, MAX_HIFF)
    ramp_up_and_down[:ramp_up.size] = ramp_up
    ramp_up_and_down[-ramp_down.size:] = ramp_down

    env = Environments(count)
    env.min_fitness.from_numpy(
        np.broadcast_to(
            np.expand_dims(ramp_up_and_down, 1),
            env.shape))
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
