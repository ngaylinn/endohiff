import numpy as np
import taichi as ti

from constants import (
    BITSTR_POWER, BITSTR_LEN,  ENVIRONMENT_SHAPE, NUM_WEIGHTS, MAX_SCORE)

@ti.data_oriented
class Environment:
    def __init__(self):
        self.min_fitness = ti.field(
            dtype=float, shape=ENVIRONMENT_SHAPE)
        self.weights = ti.Vector.field(
            n=NUM_WEIGHTS, dtype=float, shape=ENVIRONMENT_SHAPE)


def make_flat_environment():
    env = Environment()
    env.min_fitness.fill(0.0)
    env.weights.fill(1.0)
    return env


def make_random_environment():
    env = Environment()
    env.min_fitness.from_numpy(np.random.rand(*ENVIRONMENT_SHAPE) * MAX_SCORE)
    env.weights.from_numpy(np.random.rand(*ENVIRONMENT_SHAPE))
    return env


def make_designed_environment():
    # TODO: Finish implementing this.
    env_width = ENVIRONMENT_SHAPE[0]
    ramp_up_and_down = np.hstack((
        np.linspace(0.0, 1.0, 2 * env_width // 3),
        np.linspace(1.0, 0.0, 2 * env_width // 3 - 1)[1:]))
    weights = np.zeros(ENVIRONMENT_SHAPE + (NUM_WEIGHTS,))
    w = 0
    for p in range(BITSTR_POWER):
        substr_len = 2**(p+1)
        substr_weights = ...
        for _ in range(0, BITSTR_LEN, substr_len):
            weights[:, :, w] = substr_weights
