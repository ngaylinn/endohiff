"""The Environments data type and definitions for all control environments.
"""

import numpy as np
import taichi as ti

from constants import (
    BITSTR_POWER, BITSTR_LEN,  ENVIRONMENT_SHAPE, MIN_HIFF, MAX_HIFF)


def expand_shape(raw_shape=None):
    if raw_shape is None:
        shape = (1,)
    elif isinstance(raw_shape, int):
        shape = (raw_shape,)
    else:
        shape = raw_shape
        assert isinstance(shape, tuple)
        assert all(isinstance(dim, int) for dim in shape)
    if shape[-2:] != ENVIRONMENT_SHAPE:
        shape += ENVIRONMENT_SHAPE
    return shape


def make_field(shape=None):
    shape = expand_shape(shape)
    return ti.field(ti.float16, shape=shape)


def make_flat(shape=None):
    return np.zeros(shape=expand_shape(shape), dtype=np.float16)


def make_baym(shape=None):
    ew, _ = ENVIRONMENT_SHAPE
    buckets = 2 * BITSTR_POWER - 1
    bucket_width = ew // buckets
    ramp_up = np.arange(MIN_HIFF, MAX_HIFF, BITSTR_LEN).repeat(bucket_width)
    ramp_down = np.flip(ramp_up)
    ramp_up_and_down = np.full(ew, MAX_HIFF, dtype=np.float16)
    ramp_up_and_down[:ramp_up.size] = ramp_up
    ramp_up_and_down[-ramp_down.size:] = ramp_down

    shape = expand_shape(shape)
    return np.broadcast_to(np.expand_dims(ramp_up_and_down, 1), shape)


# The named environments to experiment with.
STATIC_ENVIRONMENTS = {
    'flat': make_flat,
    'baym': make_baym,
}

ALL_ENVIRONMENT_NAMES = sorted(list(STATIC_ENVIRONMENTS.keys()) + ['cppn'])


# A demo to visualize any of the environments defined above.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from visualize_inner_population import render_env_map

    ti.init(ti.cuda)

    render_env_map(STATIC_ENVIRONMENTS['baym']()[0])
    plt.show()
