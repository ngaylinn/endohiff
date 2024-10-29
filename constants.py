from pathlib import Path

import numpy as np
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

MAX_POPULATION_SIZE = np.prod(ENVIRONMENT_SHAPE) * CARRYING_CAPACITY

INNER_GENERATIONS = 100

TOURNAMENT_SIZE = 2

CROSSOVER_RATE = 0.5 #for uniform_crossover, frequency of crossover

# The chance that an cell vacated due to death will be filled in the next
# generation.
REFILL_RATE = 0.5

OUTPUT_PATH = Path('output')
