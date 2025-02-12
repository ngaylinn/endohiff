"""Compute fitness of environments from evovability of inner populations.

This class answers the question "which environment was best able to evolve an
inner population to solve hiff?" All computation happens on the GPU, and
eventually multiple different fitness criteria will be supported.
"""

from enum import Enum

import numpy as np
import taichi as ti

from constants import (
    BITSTR_LEN, CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS,
    NUM_TRIALS)


class BitstrKind(Enum):
    MORE_ZEROS = 0
    MORE_ONES = 1
    EVEN_SPLIT = 2
    OVERALL = 3


class FitnessCriteria(Enum):
    UNIFORM = 0
    DIVERSE = 1
    MIXED = 2
    ANY = 3


@ti.data_oriented
class FitnessEvaluator:
    def __init__(self, outer_population=None, criteria=FitnessCriteria.ANY):
        # If there is no outer population, then assume we've got NUM_TRIALS
        # inner_populations to work with and set up some fields that can play
        # the same role as the index and matchmaker in an outer population.
        if outer_population is None:
            self.fitness = ti.field(float, shape=(1, NUM_TRIALS, 1))
            self.index = ti.Vector.field(n=2, dtype=int, shape=NUM_TRIALS)
            self.index.from_numpy(np.array(
                list(np.ndindex(NUM_TRIALS, 1)), dtype=np.int32))
        else:
            self.fitness = outer_population.matchmaker.fitness
            self.index = outer_population.index
        self.num_environments = self.index.shape[0]

        # Separate partial sum fields for all bitstrings and ones that are
        # primarily ones, primarily zeros, an even split of both.
        self.partial_sums = ti.Vector.field(
            n=4, dtype=float, shape=self.num_environments)

        self.goal = criteria.value

    @ti.func
    def set_fitness(self, e, og, fitness):
        # Look up the trial and trial-relative index of the given environment.
        t, i = self.index[e]
        self.fitness[og, t, i] = fitness

    @ti.func
    def get_fitness(self, e, og):
        # Look up the trial and trial-relative index of the given environment.
        t, i = self.index[e]
        return self.fitness[og, t, i]

    @ti.kernel
    def compute_partial_sums(self, inner_population: ti.template(), og: int):
        ig = INNER_GENERATIONS - 1
        shape = (self.num_environments,) + ENVIRONMENT_SHAPE
        # For every location in every environment across all trials...
        for e, x, y in ti.ndrange(*shape):
            # Count up the total hiff score and the number of individuals in
            # this location, sorted by bitstring kind (mostly 0's, mostly 1's,
            # even split, and all bitstrings overall).
            hiff_sum = ti.cast(ti.Vector([0]*4), ti.uint32)
            alive_count = ti.Vector([0]*4)
            # For each individual at this location...
            for i in range(CARRYING_CAPACITY):
                individual = inner_population.pop[e, ig, x, y, i]
                # If it's alive, classify it, and add it to the appropriate
                # count(s).
                if individual.is_alive():
                    ones = individual.one_count
                    kind = -1
                    # Thresholds for classification are the top, middle, and
                    # bottom 25% of the total number of bits.
                    if ones < BITSTR_LEN // 4:
                        kind = BitstrKind.MORE_ZEROS.value
                    elif ones > 3 * BITSTR_LEN // 4:
                        kind = BitstrKind.MORE_ONES.value
                    elif (ones > 3 * BITSTR_LEN // 8 and
                          ones < 5 * BITSTR_LEN // 8):
                        kind = BitstrKind.EVEN_SPLIT.value
                    if kind != -1:
                        hiff_sum[kind] += individual.hiff
                        alive_count[kind] += 1
                    hiff_sum[BitstrKind.OVERALL.value] += individual.hiff
                    alive_count[BitstrKind.OVERALL.value] += 1
            # Compute averages for all four kinds at once, storing 0.0 for any
            # locations with no individuals of the given kind.
            self.partial_sums[e] += ti.select(
                alive_count > 0, hiff_sum / alive_count, 0.0)

    @ti.kernel
    def finalize_count(self, og: int):
        # For each environment, look at the counts of bitstrings of different
        # kinds and come up with a single fitness score.
        for e in range(self.num_environments):
            fitness = self.partial_sums[e][BitstrKind.OVERALL.value]
            # Prefer simulations where the dominant solutions are either
            # majority one or majority zero.
            if ti.static(self.goal) == FitnessCriteria.UNIFORM.value:
                fitness += max(
                    self.partial_sums[e][BitstrKind.MORE_ZEROS.value],
                    self.partial_sums[e][BitstrKind.MORE_ONES.value])
            # Prefer simulations where the dominant solutions are an even mix
            # of majority one or majority zero.
            if ti.static(self.goal) == FitnessCriteria.DIVERSE.value:
                fitness += min(
                    self.partial_sums[e][BitstrKind.MORE_ZEROS.value],
                    self.partial_sums[e][BitstrKind.MORE_ONES.value])
            # Prefer solutions where the dominant solutions are a mix of ones
            # and zeros.
            if ti.static(self.goal) == FitnessCriteria.MIXED.value:
                fitness += self.partial_sums[e][BitstrKind.EVEN_SPLIT.value]
            # Actually finalize and store the fitness score.
            self.set_fitness(e, og, fitness)

    def score_populations(self, inner_population, og):
        self.partial_sums.fill(0.0)
        self.compute_partial_sums(inner_population, og)
        self.finalize_count(og)

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
    evaluator = FitnessEvaluator(None)
    evaluator.score_populations(inner_population, 0)
    return evaluator.get_best_trial(0)


def get_per_trial_scores(inner_population):
    evaluator = FitnessEvaluator(None)
    evaluator.score_populations(inner_population, 0)
    return evaluator.fitness.to_numpy().flatten()
