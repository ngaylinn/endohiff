"""Utilities for setting up environments for evolution.
"""

import numpy as np
import taichi as ti

from src.constants import (
    BITSTR_POWER, BITSTR_LEN, ENV_SHAPE, MIN_HIFF, MAX_HIFF)


def expand_shape(raw_shape=None):
    """Expand a shorthand shape definition to canonical form.
    """
    if raw_shape is None:
        shape = (1,)
    elif isinstance(raw_shape, int):
        shape = (raw_shape,)
    else:
        shape = raw_shape
        assert isinstance(shape, tuple)
        assert all(isinstance(dim, int) for dim in shape)
    if shape[-2:] != ENV_SHAPE:
        shape += ENV_SHAPE
    return shape


def make_env_field(shape=None):
    """Allocates GPU memory to represent an environment.
    """
    shape = expand_shape(shape)
    return ti.field(ti.float16, shape=shape)


def make_flat(shape=None):
    """Generate data for the flat environment, in a given shape.

    This function makes one or many copies of the flat environment, according
    to shape. This is useful for simulating many copies of this environment in
    parallel on the GPU.
    """
    return np.zeros(shape=expand_shape(shape), dtype=np.float16)


def make_baym(shape=None):
    """Generate data for the baym environment, in a given shape.

    This function makes one or many copies of the flat environment, according
    to shape. This is useful for simulating many copies of this environment in
    parallel on the GPU.
    """
    ew, _ = ENV_SHAPE
    buckets = 2 * BITSTR_POWER - 1
    bucket_width = ew // buckets
    ramp_up = np.arange(MIN_HIFF, MAX_HIFF, BITSTR_LEN).repeat(bucket_width)
    ramp_down = np.flip(ramp_up)
    ramp_up_and_down = np.full(ew, MAX_HIFF, dtype=np.float16)
    ramp_up_and_down[:ramp_up.size] = ramp_up
    ramp_up_and_down[-ramp_down.size:] = ramp_down

    shape = expand_shape(shape)
    return np.broadcast_to(np.expand_dims(ramp_up_and_down, 1), shape)


# The named static environments to experiment with.
STATIC_ENVIRONMENTS = {
    'flat': make_flat,
    'baym': make_baym,
}


# A demo to visualize any of the environments defined above.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.visualize_one import render_env_map

    render_env_map(make_baym())
    plt.show()

