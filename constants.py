"""Constants and hyperparameters shared with all modules.
"""

from pathlib import Path

import taichi as ti

# The size of the bit strings to evolve, and several related constants.
BITSTR_POWER = 6
BITSTR_LEN = 2 ** BITSTR_POWER
BITSTR_DTYPE = ti.uint64
MIN_HIFF = BITSTR_LEN
MAX_HIFF = BITSTR_LEN * (BITSTR_POWER + 1)
NUM_WEIGHTS = BITSTR_LEN - 1

# Constants specifying the size and shape of the environment.
# A 16x9 aspect ratio is ideal for most current displays.
ENVIRONMENT_SHAPE = (64, 36)
# Visualize the population at some location as a square that's POP_TILE_SIZE on
# each side. The total number of individuals is CARRYING_CAPACITY
POP_TILE_SIZE = 5
CARRYING_CAPACITY = POP_TILE_SIZE ** 2

# Constants configuring simulated evolution
CROSSOVER_RATE = 0.5
# The probability of flipping a bit is 1/(2**MUTATION_MAGNITUDE) ~= 0.016
MUTATION_MAGNITUDE = 6
REFILL_RATE = 1.0
MIGRATION_RATE = 0.5
TOURNAMENT_SIZE = 2
INNER_GENERATIONS = 100
OUTER_GENERATIONS = 100
OUTER_POPULATION_SIZE = 50
NUM_TRIALS = 5

# Where to save all the results we generate.
OUTPUT_PATH = Path('output')
