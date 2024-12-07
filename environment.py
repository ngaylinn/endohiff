import numpy as np
import taichi as ti

from constants import (
    BITSTR_POWER, BITSTR_LEN,  ENVIRONMENT_SHAPE, NUM_WEIGHTS, MIN_HIFF,
    MAX_HIFF)


@ti.data_oriented
class Environment:
    def __init__(self):
        self.min_fitness = ti.field(
            dtype=float, shape=ENVIRONMENT_SHAPE)
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


def make_nate1():
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
        weights[:, :, w:w+substr_count] = substr_weights
        w += substr_count

    # The min fitness threshold simply scales linearly from the top to the
    # bottom of the environment, creating a smooth gradient of habitibality
    # across the full range of weights.
    min_fitness = np.broadcast_to(
        np.geomspace(1, MAX_HIFF, eh, dtype=np.float32),
        ENVIRONMENT_SHAPE)

    env = Environment()
    env.min_fitness.from_numpy(min_fitness)
    env.weights.from_numpy(weights)
    return env


def make_nate2():
    weights = np.zeros(ENVIRONMENT_SHAPE + (NUM_WEIGHTS,), dtype=np.float32)
    w = 0
    for p in range(BITSTR_POWER):
        substr_len = 2**(p + 1)
        substr_count = BITSTR_LEN // substr_len
        substr_weights = np.eye(substr_count).repeat(substr_len, axis=0)
        weights[:, :, w:w+substr_count] = np.expand_dims(
            substr_weights, 1
        ).repeat(ENVIRONMENT_SHAPE[1], axis=1)
        w += substr_count

    env = Environment()
    env.min_fitness.fill(0.0)
    env.weights.from_numpy(weights)
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


ENVIRONMENTS = {
    'flat': make_flat,
    'random': make_random,
    'baym': make_baym,
}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from render import render_env_map

    ti.init(ti.cuda, unrolling_limit=0)

    env = make_nate2()
    render_env_map(env.to_numpy())
    plt.tight_layout()
    plt.show()
