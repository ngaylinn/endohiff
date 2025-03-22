"""Compute fitness of environments from evovability of inner populations.

This class answers the question "which environment was best able to evolve an
inner population to solve hiff?" All computation happens on the GPU, and
eventually multiple different fitness criteria will be supported.
"""
import numpy as np
import taichi as ti

from constants import (
    CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS, NUM_TRIALS)


@ti.data_oriented
class FitnessEvaluator:
    def __init__(self, count=1, outer_population=None):
        # If there is no outer population, then assume we've got NUM_TRIALS
        # inner_populations to work with and set up some fields that can play
        # the same role as the index and matchmaker in an outer population.
        if outer_population is None:
            self.fitness = ti.field(float, shape=(1, count, 1))
            self.index = ti.Vector.field(n=2, dtype=int, shape=count)
            self.index.from_numpy(np.array(
                list(np.ndindex(count, 1)), dtype=np.int32))
        else:
            self.fitness = outer_population.matchmaker.fitness
            self.index = outer_population.index
        self.num_environments = self.index.shape[0]

    @ti.func
    def inc_fitness(self, e, og, fitness):
        # Look up the trial and trial-relative index of the given environment.
        t, i = self.index[e]
        self.fitness[og, t, i] += fitness

    @ti.func
    def get_fitness(self, e, og):
        # Look up the trial and trial-relative index of the given environment.
        t, i = self.index[e]
        return self.fitness[og, t, i]

    @ti.kernel
    def score_populations(self, inner_population: ti.template(), og: int):
        ig = INNER_GENERATIONS - 1
        shape = (self.num_environments,) + ENVIRONMENT_SHAPE
        # For every location in every environment across all trials...
        for e, x, y in ti.ndrange(*shape):
            # Count all the individuals here and their fitness.
            alive_count = 0.0
            fitness_sum = 0.0
            for i in range(CARRYING_CAPACITY):
                individual = inner_population.pop[e, ig, x, y, i]
                if individual.is_alive():
                    alive_count += 1
                    fitness_sum += individual.fitness
            # Compute averages for all four kinds at once, storing 0.0 for any
            # locations with no living individuals.
            local_fitness = ti.select(
                alive_count > 0, fitness_sum / alive_count, 0.0)
            self.inc_fitness(e, og, local_fitness)

    @ti.kernel
    def get_best_per_trial(self, og: int) -> ti.types.vector(n=NUM_TRIALS, dtype=int):
        best_index = ti.Vector([-1] * NUM_TRIALS)
        best_fitness = ti.Vector([-1.0] * NUM_TRIALS)
        for e in range(self.num_environments):
            t = self.index[e][0]
            fitness = self.get_fitness(e, og)
            # Set the max fitness for this trial in a thread-safe way, then
            # check to see if this was the thread that won and store the
            # associated index if so.
            ti.atomic_max(best_fitness[t], fitness)
            if fitness == best_fitness[t]:
                best_index[t] = e
        return best_index

    @ti.kernel
    def get_best_trial(self, og: int) -> int:
        best_index = -1
        best_fitness = -1.0
        for e in range(self.num_environments):
            fitness = self.get_fitness(e, og)
            # Set the max fitness in a thread-safe way, then check to see if
            # this was the thread that won and store the associated index if
            # so.
            ti.atomic_max(best_fitness, fitness)
            if fitness == best_fitness:
                best_index = e

        # Look up the trial corresponding to the best environment of all.
        return self.index[best_index][0]


def get_best_trial(inner_population):
    """A utility for quickly scoring an inner_population without an outer one.
    """
    evaluator = FitnessEvaluator(inner_population.shape[0])
    evaluator.score_populations(inner_population, 0)
    return evaluator.get_best_trial(0)


def get_per_trial_scores(inner_population):
    evaluator = FitnessEvaluator(inner_population.shape[0])
    evaluator.score_populations(inner_population, 0)
    return evaluator.fitness.to_numpy().flatten()
