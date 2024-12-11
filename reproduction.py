"""Functions for breeding individuals from the inner population.
"""

import taichi as ti

from constants import (
    BITSTR_DTYPE, BITSTR_LEN, CARRYING_CAPACITY, CROSSOVER_RATE, DEAD_ID,
    MUTATION_MAGNITUDE, TOURNAMENT_SIZE)


@ti.func
def mutation() -> BITSTR_DTYPE:
    # Start with a bitstring of all ones, then repeatedly generate random
    # bistrings and combine them using bitwise and. Each bit in the final
    # result will be 1 if and only if it was randomly chosen to be 1 every
    # time. Since each bit has a 1/2 probability of being 1 in each iteration,
    # the final probability of each bit being set is 1/(2**MUTATION_MAGNITUDE)
    mutation = ti.cast(-1, BITSTR_DTYPE)
    for _ in range(MUTATION_MAGNITUDE):
        mutation &= ti.random(ti.uint64)
    return mutation


@ti.func
def crossover(bitstr1: BITSTR_DTYPE, bitstr2: BITSTR_DTYPE) -> BITSTR_DTYPE:
    # Do one-point crossover on the two bit strings.
    crossover_point = ti.random(int) % BITSTR_LEN
    mask = (1 << crossover_point) - 1
    return (bitstr1 & mask) | (bitstr2 & ~mask)


@ti.func
def tournament_selection(pop: ti.template(), g: int, x: int, y: int) -> ti.template():
    # If there are no living individuals at this location, return -1.
    best_index = -1
    best_fitness = -float('inf')
    if ti.random() < CROSSOVER_RATE:
        # Compare TOURNAMENT_SIZE random individuals
        for _ in range(TOURNAMENT_SIZE):
            # There may be dead individuals in this population, so we have to
            # keep searching until we find a potential living mate.
            offset = ti.random(int)
            for i in range(CARRYING_CAPACITY):
                competitor_index = (i + offset % CARRYING_CAPACITY)
                competitor = pop[g, x, y, competitor_index]
                if competitor.id != DEAD_ID and competitor.fitness > best_fitness:
                    best_index = competitor_index
                    best_fitness = competitor.fitness

    return best_index
