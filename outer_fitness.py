"""Compute fitness of environments from evovability of inner populations.

This class answers the question "which environment was best able to evolve an
inner population to solve hiff?" All computation happens on the GPU, and
eventually multiple different fitness criteria will be supported.
"""

import taichi as ti

from constants import (
    CARRYING_CAPACITY, DEAD_ID, ENVIRONMENT_SHAPE, INNER_GENERATIONS,
    OUTER_GENERATIONS)


@ti.data_oriented
class FitnessEvaluator:
    def __init__(self, outer_population, num_environments=None):
        # We want to use the same fitness criteria for evolving an inner
        # population in a static environment or within an outer population of
        # evolved environments. This means both need to happen on the GPU, but
        # we need to allocate a field for the results if there wasn't a
        # matchmaker already allocated for the outer population.
        if outer_population is None:
            self.use_matchmaker = False
            assert isinstance(num_environments, int)
            self.num_environments = num_environments
            self.fitness = ti.field(float, num_environments)
        else:
            self.use_matchmaker = True
            self.num_environments = outer_population.index.shape[0]
            self.outer_population = outer_population

    @ti.func
    def inc_fitness(self, e, og, fitness):
        if ti.static(self.use_matchmaker):
            sp, i = self.outer_population.index[e]
            self.outer_population.matchmaker.fitness[og, sp, i] += fitness
        else:
            self.fitness[e] += fitness

    @ti.func
    def get_fitness(self, e, og):
        if ti.static(self.use_matchmaker):
            sp, i = self.outer_population.index[e]
            return self.outer_population.matchmaker.fitness[og, sp, i]
        else:
            return self.fitness[e]

    @ti.kernel
    def score_populations(self, inner_population: ti.template(), og: int):
        ig = INNER_GENERATIONS - 1
        shape = (self.num_environments,) + ENVIRONMENT_SHAPE
        for e, x, y in ti.ndrange(*shape):
            hiff_sum = ti.cast(0, ti.uint32)
            alive_count = 0
            for i in range(CARRYING_CAPACITY):
                individual = inner_population.pop[e, ig, x, y, i]
                if individual.id != DEAD_ID:
                    hiff_sum += individual.hiff
                    alive_count += 1
            mean_hiff = ti.select(alive_count > 0, hiff_sum / alive_count, 0.0)
            self.inc_fitness(e, og, mean_hiff ** 2)

    @ti.kernel
    def get_best_kernel(self) -> ti.math.vec2:
        og = OUTER_GENERATIONS - 1
        best_index = -1
        best_fitness = -1.0
        for e in range(self.num_environments):
            fitness = self.get_fitness(e, og)
            if fitness > best_fitness:
                best_fitness = fitness
                best_index = e
        return ti.Vector([float(best_index), best_fitness])

    def get_best(self):
        best_index, best_fitness = self.get_best_kernel()
        return int(best_index), best_fitness


def get_best(inner_population):
    """A utility for quickly scoring an inner_population without an outer one.
    """
    evaluator = FitnessEvaluator(
        outer_population=None,
        num_environments=inner_population.num_environments)
    evaluator.score_populations(inner_population, -1)
    return evaluator.get_best()
