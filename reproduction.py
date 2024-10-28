import taichi as ti

from constants import (
    BITSTR_DTYPE, BITSTR_LEN, CARRYING_CAPACITY, TOURNAMENT_SIZE)


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

@ti.func
def crossover(parent1: BITSTR_DTYPE, parent2: BITSTR_DTYPE) -> BITSTR_DTYPE:
    crossover_point = ti.random(ti.i32) % BITSTR_LEN
    mask = (1 << crossover_point) - 1
    child = (parent1 & mask) | (parent2 & ~mask)
    return child

@ti.func
def tournament_selection(pop: ti.template(), g: int, x: int, y: int) -> ti.template():
    """Performs a simple tournament selection within a sub-population"""
    best_individual = pop[g, x, y, 0]
    best_fitness = -float('inf')

    # Compare TOURNAMENT_SIZE random individuals
    for _ in range(TOURNAMENT_SIZE):
        competitor_index = ti.random(ti.i32) % CARRYING_CAPACITY
        competitor = pop[g, x, y, competitor_index]

        if competitor.fitness > best_fitness:
            best_individual = competitor
            best_fitness = competitor.fitness

    return best_individual
