import taichi as ti

BITSTR_POWER = 6
BITSTR_LEN = 2 ** BITSTR_POWER
BITSTR_DTYPE = ti.uint64
NUM_WEIGHTS = BITSTR_LEN - 1
MAX_SCORE = BITSTR_LEN * (BITSTR_POWER + 1)

# A 16x9 aspect ratio is ideal for most current displays.
ENVIRONMENT_SHAPE = (64, 36)
# A square number makes visualizing results easier.
CARRYING_CAPACITY = 25

NUM_GENERATIONS = 100
