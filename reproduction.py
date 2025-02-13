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


@ti.data_oriented
class TournamentArena:
    """Manages memory allocations for tournament selection."""
    def __init__(self, pop):
        # This class is designed to make one selection for every individual in
        # a population.
        self.pop = pop
        ne, ig, ew, eh, cc = pop.shape
        # What individuals have already been considered in the tournament.
        self.seen = ti.field(bool, (ne, ew, eh, cc, cc))
        # What selections were made
        self.selections = ti.field(int, (ne, ew, eh, cc))

    @ti.kernel
    def select_all(self, g: int, params: ti.template()):
        self.seen.fill(False)
        for e, x, y, i in ti.ndrange(*self.selections.shape):
            self.selections[e, x, y, i] = self.select_one(
                e, g, x, y, i, params[e].tournament_size)

    @ti.func
    def select_one(self, e, g, x, y, i, tournament_size):
        # If there are no living individuals at this location, return -1.
        best_index = -1
        best_fitness = -1.0

        for _ in range(tournament_size):
            # For each tournament, go through the population at this location
            # starting from a randomly chosen offset.
            # NOTE: Careful typing to satisfy Taichi's debugger.
            offset = ti.cast(ti.random(ti.uint32) % CARRYING_CAPACITY, ti.int32)
            for ci in range(CARRYING_CAPACITY):
                c = (ci + offset) % CARRYING_CAPACITY

                # We should look at each individual at most once. This way, a
                # tournament size of CARRYING_CAPCITY means we check everyone.
                if self.seen[e, x, y, i, c]:
                    continue

                # Any living individual we find is a candidate for selection.
                candidate = self.pop[e, g, x, y, c]
                if candidate.is_alive():

                    # Keep track of the best one, who will be the winner.
                    if candidate.fitness > best_fitness:
                        best_index = c
                        best_fitness = candidate.fitness

                    # Remember we considered this candidate (even if they're
                    # not the best) and move onto the next round.
                    self.seen[e, x, y, i, c] = True
                    break

        return best_index


def validate_selection():
    """A sanity check to make sure selection is working right.
    """
    import numpy as np

    from inner_population import Individual, get_default_params

    # Do selection NUM_TESTS times with all possible values of tournament size
    # (that is, from 1 to CARRYING_CAPACITY, inclusive).
    NUM_TESTS = 10_000
    ne = CARRYING_CAPACITY
    ig = 1
    ew = NUM_TESTS
    eh = 1
    cc = CARRYING_CAPACITY

    # Set up NUM_TESTS populations of CARRYING_CAPACITY individuals, with
    # fitness values 1..CARRYING_CAPACITY.
    pop = Individual.field(shape=(ne, ig, ew, eh, cc))
    pop.fitness.from_numpy(
        np.tile(
            np.arange(CARRYING_CAPACITY), ew * ne
        ).reshape((ne, ig, ew, eh, cc)))

    # Each of the NUM_TESTS populations gets assigned one of the possible
    # values for tournament size, ranging from 1 to CARRYING_CAPACITY.
    params = get_default_params(CARRYING_CAPACITY)
    params.tournament_size.from_numpy(
        np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1)

    # Do selection on all the populations in parallel!
    arena = TournamentArena(pop)
    arena.select_all(0, params)

    # Print a summary of results. When tournament_size == 1, you should get ~13
    # which indicates we picked the median individual (rounded up). When
    # tournament_size == CARRYING_CAPACITY, you should get exactly
    # CARRYING_CAPACITY, indicating we found the best individual.
    print(arena.selections.to_numpy().shape)
    mean_selection = (
        arena.selections.to_numpy() + 1
    ).mean(axis=(1, 2, 3))
    print(mean_selection.shape)
    print('Mean selection for tournament of size:')
    for i in range(CARRYING_CAPACITY):
        print(i + 1, f'{mean_selection[i]:0.3f}')


if __name__ == '__main__':
    ti.init(ti.cuda, debug=True)
    validate_selection()
