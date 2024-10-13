from pathlib import Path

import taichi as ti

BITSTR_POWER = 6
BITSTR_LEN = 2 ** BITSTR_POWER
BITSTR_DTYPE = ti.uint64
NUM_WEIGHTS = BITSTR_LEN - 1
MIN_HIFF = BITSTR_LEN
MAX_HIFF = BITSTR_LEN * (BITSTR_POWER + 1)

# A 16x9 aspect ratio is ideal for most current displays.
ENVIRONMENT_SHAPE = (64, 36)
# A square number makes visualizing results easier.
CARRYING_CAPACITY = 25

INNER_GENERATIONS = 100
OUTER_GENERATIONS = 25

# TODO: How many do you actually need for statistical significance?
NUM_TRIALS = 5

NUM_HOST_ENVIRONMENTS = NUM_TRIALS * CARRYING_CAPACITY
ALL_ENVIRONMENTS_SHAPE = (NUM_HOST_ENVIRONMENTS,) + ENVIRONMENT_SHAPE

OUTPUT_PATH = Path('output')
