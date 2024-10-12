import taichi as ti

from constants import BITSTR_DTYPE


# The probability of flipping a bit is 1/(2**MUTATION_MAGNITUDE) ~= 0.016
MUTATION_MAGNITUDE = 6

@ti.func
def mutation() -> BITSTR_DTYPE:
    # Start with a bitstring of all ones, then repeatedly generate random
    # bistrings and combine them using bitwise and. Each bit in the final
    # result will be 1 if and only if it was randomly chosen to be 1 every
    # time. Since each bit has a 1/2 probability of being 1 in each iteration,
    # the final probability of each bit being set is 1/(2**MUTATION_MAGNITUDE)
    mutation = ti.cast(-1, BITSTR_DTYPE)
    for _ in range(MUTATION_MAGNITUDE):
        mutation &= ti.cast(ti.random(int), BITSTR_DTYPE)
    return mutation



