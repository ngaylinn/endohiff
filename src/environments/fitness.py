import numpy as np
import taichi as ti

from ..constants import (
    CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS, MAX_HIFF,
    NUM_TRIALS)

MAX_OUTER_FITNESS = MAX_HIFF + 1


@ti.data_oriented
class EnvironmentFitnessEvaluator:
    def __init__(self, count=1, env_pop=None):
        # If there is no outer population, then assume we've got count
        # bitstr_pops to work with and set up some fields that can play
        # the same role as the index and matchmaker in an outer population.
        if env_pop is None:
            self.fitness = ti.field(float, shape=(1, count, 1))
            self.index = ti.Vector.field(n=2, dtype=int, shape=count)
            self.index.from_numpy(np.array(
                list(np.ndindex(count, 1)), dtype=np.int32))
        else:
            self.fitness = env_pop.matchmaker.fitness
            self.index = env_pop.index
        self.num_environments = self.index.shape[0]

    @ti.kernel
    def get_max_score(self, bitstr_pop: ti.template(), og: int):
        shape = (self.num_environments, INNER_GENERATIONS) + ENVIRONMENT_SHAPE
        # For every location in every environment across all trials and
        # generations...
        for e, ig, x, y in ti.ndrange(*shape):
            t, oi = self.index[e]
            local_max_fitness = 0.0

            # Find the most fit individual in this local population and compare
            # that to the max score for this environment.
            for ii in ti.ndrange(INNER_GENERATIONS, CARRYING_CAPACITY):
                individual = bitstr_pop.pop[e, ig, x, y, ii]
                if individual.is_alive():
                    local_max_fitness = max(local_max_fitness,
                                            individual.fitness)
            ti.atomic_max(self.fitness[og, t, oi], local_max_fitness)

    @ti.kernel
    def get_first_instance(self, bitstr_pop: ti.template(), og: int):
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
                    individual = bitstr_pop.pop[e, ig, x, y, ii]
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

    def score_populations(self, bitstr_pop, og):
        # TODO: This is very inefficient. Maybe find a better way?
        self.get_max_score(bitstr_pop, og)
        self.get_first_instance(bitstr_pop, og)

    @ti.kernel
    def get_best_per_trial(self, og: int) -> ti.types.vector(n=NUM_TRIALS, dtype=int):
        best_index = ti.Vector([-1] * NUM_TRIALS)
        best_fitness = ti.Vector([-1.0] * NUM_TRIALS)
        for e in range(self.num_environments):
            t, i = self.index[e]
            fitness = self.fitness[og, t, i]
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
            t, i = self.index[e]
            fitness = self.fitness[og, t, i]
            # Set the max fitness in a thread-safe way, then check to see if
            # this was the thread that won and store the associated index if
            # so.
            ti.atomic_max(best_fitness, fitness)
            if fitness == best_fitness:
                best_index = e

        # Look up the trial corresponding to the best environment of all.
        return self.index[best_index][0]


def get_best_trial(bitstr_pop):
    """A utility for quickly scoring an bitstr_pop without an outer one.
    """
    evaluator = EnvironmentFitnessEvaluator(bitstr_pop.shape[0])
    evaluator.score_populations(bitstr_pop, 0)
    return evaluator.get_best_trial(0)


def get_per_trial_scores(bitstr_pop):
    evaluator = EnvironmentFitnessEvaluator(bitstr_pop.shape[0])
    evaluator.score_populations(bitstr_pop, 0)
    return evaluator.fitness.to_numpy().flatten()

