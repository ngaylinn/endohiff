"""Compute fitness of environments from evovability of inner populations.

This class answers the question "which environment was best able to evolve an
inner population to solve hiff?" All computation happens on the GPU, and
eventually multiple different fitness criteria will be supported.
"""
import numpy as np
import taichi as ti

from constants import (
    CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS, MAX_HIFF,
    NUM_TRIALS)

MAX_OUTER_FITNESS = MAX_HIFF + 1


@ti.data_oriented
class FitnessEvaluator:
    def __init__(self, count=1, outer_population=None):
        # If there is no outer population, then assume we've got count
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
    def get_max_score(self, inner_population: ti.template(), og: int):
        shape = (self.num_environments, INNER_GENERATIONS) + ENVIRONMENT_SHAPE
        # For every location in every environment across all trials and
        # generations...
        for e, ig, x, y in ti.ndrange(*shape):
            t, oi = self.index[e]
            local_max_fitness = 0.0

            # Find the most fit individual in this local population and compare
            # that to the max score for this environment.
            for ii in ti.ndrange(INNER_GENERATIONS, CARRYING_CAPACITY):
                individual = inner_population.pop[e, ig, x, y, ii]
                if individual.is_alive():
                    local_max_fitness = max(local_max_fitness,
                                            individual.fitness)
            ti.atomic_max(self.fitness[og, t, oi], local_max_fitness)

    @ti.kernel
    def get_first_instance(self, inner_population: ti.template(), og: int):
        shape = ENVIRONMENT_SHAPE + (CARRYING_CAPACITY,)
        # For every environment across all trials...
        for e in range(self.num_environments):
            t, oi = self.index[e]
            global_max_fitness = self.fitness[og, t, oi]

            # For each generation...
            for ig in range(INNER_GENERATIONS):
                # Find the best fitness score in this generation.
                local_max_fitness = 0.0
                for x, y, ii in ti.ndrange(*shape):
                    individual = inner_population.pop[e, ig, x, y, ii]
                    if individual.is_alive():
                        local_max_fitness = max(local_max_fitness,
                                                individual.fitness)
                # If it matches the global max score, then this is when the
                # global max score first arose. Adjust the score accordingly.
                if local_max_fitness == global_max_fitness:
                    earliness = (INNER_GENERATIONS - ig) / INNER_GENERATIONS
                    # HIFF is an integral fitness function so this earliness
                    # factor only serves to break ties.
                    self.fitness[og, t, oi] = global_max_fitness + earliness
                    break

    def score_populations(self, inner_population, og):
        self.get_max_score(inner_population, og)
        self.get_first_instance(inner_population, og)

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
