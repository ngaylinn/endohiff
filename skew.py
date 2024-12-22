"""Verify there's no preference for 0s or 1s when evolving to solve hiff.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
import seaborn as sns
import taichi as ti
from tqdm import trange

from constants import BITSTR_LEN, CARRYING_CAPACITY, DEAD_ID, ENVIRONMENT_SHAPE, INNER_GENERATIONS
from environment import ENVIRONMENTS
from inner_population import InnerPopulation

ti.init(ti.cuda)


# Set these as high as you like to improve statistical significance.
# NUM_PARALELL determines how many trials to run at once on the GPU. Set this
# as high as your GPU can handle to maximize performance.
TOTAL_TRIALS = 1000
NUM_PARALLEL = 20

# Track how many ones each evolved bitstring has. Since each item is a value
# from 0 to 64 it would be much more efficient to make a histogram of how often
# each of those values appears, but then we'd need a different method of
# computing skew.
counts = ti.field(
    ti.int8, (TOTAL_TRIALS,) + ENVIRONMENT_SHAPE + (CARRYING_CAPACITY,))


@ti.func
def count_ones(bitstr):
    ones = 0
    for b in range(BITSTR_LEN):
        ones += int((bitstr >> b) & 1)
    return ones


@ti.kernel
def update_totals(population: ti.template(), trials_completed: int):
    g = INNER_GENERATIONS - 1
    for e, x, y, i in ti.ndrange(*population.shape):
        individual = population.pop[e, g, x, y, i]
        # Each environment is a distinct trial on top of the ones completed so
        # far, so compute the absolute trial number for this individual.
        t = trials_completed + e
        if individual.id == DEAD_ID:
            counts[t, x, y, i] = ti.cast(-1, ti.int8)
        else:
            one_count = count_ones(individual.bitstr)
            counts[t, x, y, i] = ti.cast(one_count, ti.int8)


def evaluate_skew(env, migration, crossover):
    inner_population = InnerPopulation(NUM_PARALLEL)
    environment = ENVIRONMENTS[env](NUM_PARALLEL)

    # Evolve a whole population TOTAL_TRIALS times running many trials in
    # paralell on the GPU.
    for i in trange(TOTAL_TRIALS // NUM_PARALLEL):
        inner_population.randomize()
        for inner_generation in range(INNER_GENERATIONS):
            inner_population.evaluate(environment, inner_generation)

            if inner_generation + 1 < INNER_GENERATIONS:
                inner_population.propagate(
                    environment, inner_generation, migration, crossover)
        # Count up the number of 1's in each evolved individual.
        update_totals(inner_population, i * NUM_PARALLEL)

    # Grab the one counts off the GPU and convert them to a flat list with DEAD
    # individuals removed.
    count_data = counts.to_numpy().flatten()
    count_data = count_data[np.where(count_data > -1)]

    # Show a histogram for visual confirmation
    sns.displot(x=count_data, kind='hist', discrete=True)
    plt.show()

    # Print some stats about the distribution.
    print(f'Median == {np.median(count_data):0.5f} (32 is expected)')
    print(f'Skew == {skew(count_data):0.5f} (0 is symmetrical)')


if __name__ == '__main__':
    evaluate_skew('baym', True, True)
