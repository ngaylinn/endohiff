"""Functions for breeding individuals from the inner population.
"""

import taichi as ti

from constants import (
    BITSTR_DTYPE, BITSTR_LEN, CARRYING_CAPACITY, CROSSOVER_RATE,
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
        mutation &= ti.cast(ti.random(int), BITSTR_DTYPE)
    return mutation


@ti.func
def crossover(bitstr1: BITSTR_DTYPE, bitstr2: BITSTR_DTYPE) -> BITSTR_DTYPE:
    # Do one-point crossover on the two bit strings.
    crossover_point = ti.random(int) % BITSTR_LEN
    mask = (1 << crossover_point) - 1
    return (bitstr1 & mask) | (bitstr2 & ~mask)


@ti.func
def tournament_selection(pop: ti.template(), g: int, x: int, y: int) -> ti.template():
    """Performs a simple tournament selection within a sub-population.
    """
    best_individual = pop[g, x, y, 0]
    best_fitness = -float('inf')

    # Compare TOURNAMENT_SIZE random individuals
    for _ in range(TOURNAMENT_SIZE):
        competitor_index = ti.random(int) % CARRYING_CAPACITY
        competitor = pop[g, x, y, competitor_index]

        if competitor.fitness > best_fitness:
            best_individual = competitor
            best_fitness = competitor.fitness

    return best_individual
