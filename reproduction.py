"""Functions for breeding individuals from the inner population.
"""

import taichi as ti

from constants import (
    BITSTR_DTYPE, BITSTR_LEN, CARRYING_CAPACITY, MUTATION_MAGNITUDE)


@ti.func
def mutation() -> BITSTR_DTYPE:
    # Start with a bitstring of all ones, then repeatedly generate random
    # bistrings and combine them using bitwise and. Each bit in the final
    # result will be 1 if and only if it was randomly chosen to be 1 every
    # time. Since each bit has a 1/2 probability of being 1 in each iteration,
    # the final probability of each bit being set is 1/(2**MUTATION_MAGNITUDE)
    mutation = ti.cast(-1, BITSTR_DTYPE)
    for _ in range(MUTATION_MAGNITUDE):
        mutation &= ti.random(BITSTR_DTYPE)
    return mutation


@ti.func
def crossover(bitstr1: BITSTR_DTYPE, bitstr2: BITSTR_DTYPE) -> BITSTR_DTYPE:
    # Do one-point crossover on the two bit strings.
    crossover_point = ti.random(ti.uint32) % BITSTR_LEN
    mask = (ti.cast(1, ti.uint64) << crossover_point) - 1
    return (bitstr1 & mask) | (bitstr2 & ~mask)


@ti.func
def tournament_selection(pop, e, g, x, y, tournament_size):
    # If there are no living individuals at this location, return -1.
    best_index = -1
    best_fitness = -1.0

    for _ in range(tournament_size):
        # For each tournament, go through the population at this location
        # starting from a randomly chosen offset.
        # NOTE: Careful typing to satisfy Taichi's debugger.
        offset = ti.cast(ti.random(ti.uint32) % CARRYING_CAPACITY, ti.int32)
        for i in range(CARRYING_CAPACITY):
            c = (i + offset) % CARRYING_CAPACITY

            # Any living individual we find is a candidate for selection.
            candidate = pop[e, g, x, y, c]
            if candidate.is_alive():
                # Keep track of the best one, who will be the winner.
                if candidate.fitness > best_fitness:
                    best_index = c
                    best_fitness = candidate.fitness
                # Move on to the next round of the tournament, even if this
                # candidate was not the winner.
                break

    return best_index
