"""Constants and hyperparameters shared by all modules.
"""

import taichi as ti

# The size of the bit strings to evolve, and several related constants.
BITSTR_POWER = 6
BITSTR_LEN = 2 ** BITSTR_POWER
BITSTR_DTYPE = ti.uint64
MIN_HIFF = BITSTR_LEN
MAX_HIFF = BITSTR_LEN * (BITSTR_POWER + 1)

# A 16x9 aspect ratio is ideal for most current displays.
ENV_SHAPE = (64, 36)
# Visualize the population at some location as a square that's POP_TILE_SIZE on
# each side. The total number of individuals is CARRYING_CAPACITY
POP_TILE_SIZE = 5
CARRYING_CAPACITY = POP_TILE_SIZE ** 2

# The probability of flipping a bit is 1/(2**MUTATION_MAGNITUDE) ~= 0.016
MUTATION_MAGNITUDE = 6
BITSTR_GENERATIONS = 150
ENV_GENERATIONS = 20
ENV_POPULATION_SIZE = 30
NUM_TRIALS = 5

ENV_NAMES = ['baym', 'flat', 'cppn']
