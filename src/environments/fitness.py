"""Evaluates the fitness of an environment from bitstrings evolved inside it.
"""

import numpy as np
import taichi as ti

from src.constants import (
    CARRYING_CAPACITY, ENV_SHAPE, ENV_GENERATIONS, MAX_HIFF)

MAX_ENV_FITNESS = MAX_HIFF + 1


@ti.kernel
def eval_env_fitness(fitness: ti.template(), index: ti.template(),
                     bitstr_pop: ti.template(), og: int):
    """Evaluate the fitness of an EnvPopulation on the GPU.
    """
    # For every location in every environment across all trials and
    # generations...
    shape = (index.shape[0], ENV_GENERATIONS) + ENV_SHAPE
    for e, ig, x, y in ti.ndrange(*shape):
        t, ei = index[e]
        local_max_fitness = 0.0

        # Find the most fit individual in this deme and compare that to the max
        # score for this environment.
        for bi in range(CARRYING_CAPACITY):
            individual = bitstr_pop.pop[e, ig, x, y, bi]
            if individual.is_alive():
                local_max_fitness = max(local_max_fitness,
                                        individual.fitness)

        # Add some "bonus points" proportional to how early in bitstring
        # evolution this high score occurred. Bitstring fitness is always an
        # integer value, and the earliness score is a number betwee 0 and 1, so
        # earliness only serves to break ties.
        earliness = (ENV_GENERATIONS - ig) / ENV_GENERATIONS
        ti.atomic_max(fitness[og, t, ei], local_max_fitness + earliness)


def get_per_trial_env_fitness(bitstr_pop):
    """Evalute the fitness of a static environment.

    This function takes bitstring populations from several trials, and computes
    the environmental fitness score for each one using the same code used on
    evolved environments above.
    """
    # This function is used for evaluating the performance of static
    # environments, and doesn't really need to run on the CPU. However, we want
    # to use exactly the same code used to evaluate the evolved environments,
    # and that has good reason to run on the GPU.
    count = bitstr_pop.shape[0]
    fitness = ti.field(float, shape=(1, count, 1))
    index = ti.Vector.field(n=2, dtype=int, shape=count)
    index.from_numpy(np.stack((
        np.arange(count, dtype=np.int32),
        np.zeros(count, dtype=np.int32))).T)
    eval_env_fitness(fitness, index, bitstr_pop, 0)
    return fitness.to_numpy().flatten()
