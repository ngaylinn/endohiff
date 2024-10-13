import numpy as np
import taichi as ti

from constants import (
    ALL_ENVIRONMENTS_SHAPE, BITSTR_POWER, BITSTR_LEN,  ENVIRONMENT_SHAPE,
    NUM_WEIGHTS, MAX_HIFF)


@ti.data_oriented
class Environment:
    def __init__(self):
        self.min_fitness = ti.field(dtype=float, shape=ALL_ENVIRONMENTS_SHAPE)
        self.weights = ti.Vector.field(
            n=NUM_WEIGHTS, dtype=float, shape=ALL_ENVIRONMENTS_SHAPE)

    def to_numpy(self):
        return {
            'min_fitness': self.min_fitness.to_numpy(),
            'weights': self.weights.to_numpy()
        }


def make_flat_environment():
    env = Environment()
    env.min_fitness.fill(0.0)
    env.weights.fill(1.0)
    return env


def make_random_environment():
    env = Environment()
    env.min_fitness.from_numpy(
        np.random.rand(*ALL_ENVIRONMENTS_SHAPE).astype(np.float32) * MAX_HIFF)
    env.weights.from_numpy(
        np.random.rand(*ALL_ENVIRONMENTS_SHAPE, NUM_WEIGHTS).astype(np.float32))
    return env


def make_designed_environment():
    ew, eh = ENVIRONMENT_SHAPE
    ramp_width = ew / (BITSTR_POWER - 1)
    ramp_up_and_down = np.hstack((
        np.zeros(ew - int(ramp_width)),
        np.linspace(0.0, 1.0, int(ramp_width)),
        np.linspace(1.0, 0.0, int(ramp_width)),
        np.zeros(ew - int(ramp_width))))
    weights = np.zeros(ENVIRONMENT_SHAPE + (NUM_WEIGHTS,), dtype=np.float32)
    w = 0
    for p in range(BITSTR_POWER):
        substr_len = 2**(p + 1)
        substr_count = BITSTR_LEN // substr_len
        start = int(ew - p * ramp_width)
        # We use the same weight for all substrings of the same size. We
        # gradually increase than decrease the weights across the width of the
        # environment, using the same values across the full height of the
        # environment. The max weight for each substring length happens at a
        # different point along the width, emphasizing short strings on the
        # left and gradually only counting the full bitstring at the far right
        # hand side.
        substr_weights = np.expand_dims(
            np.expand_dims(
                ramp_up_and_down[start:start + ew],
                axis=1
            ).repeat(eh, axis=1),
            axis=2
        ).repeat(substr_count, axis=2)
        weights[:, :, w:w+substr_count] = np.broadcast_to(
            substr_weights, ALL_ENVIRONMENTS_SHAPE)
        w += substr_count

    # The min fitness threshold simply scales linearly from the top to the
    # bottom of the environment, creating a smooth gradient of habitibality
    # across the full range of weights.
    min_fitness = np.broadcast_to(
        np.linspace(0.0, MAX_HIFF, eh, dtype=np.float32),
        ALL_ENVIRONMENTS_SHAPE)

    env = Environment()
    env.min_fitness.from_numpy(min_fitness)
    env.weights.from_numpy(weights)
    return env


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from render import render_env_map

    ti.init(ti.cuda, unrolling_limit=0)

    env = make_designed_environment()
    render_env_map(env.to_numpy())
    plt.show()
