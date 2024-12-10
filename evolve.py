"""Simulate evolution of a symbiont population in a spatial environment.
"""

import numpy as np
import polars as pl

from constants import CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS


# Compute an index enumerating the generation number and position of each
# individual in a population over evolutionary time. This will be combined with
# raw data from the simulator to produce log files.
g = INNER_GENERATIONS
w, h = ENVIRONMENT_SHAPE
c = CARRYING_CAPACITY
index = pl.DataFrame({
    'Generation': np.arange(g).repeat(w * h * c),
    'x': np.tile(np.arange(w).repeat(h * c), g),
    'y': np.tile(np.arange(h).repeat(c), g * w),
})


def outer_fitness(inner_log):
    """Score how well the symbionts evolved in this configuration.

    This is used to find the "best" run across multiple trials, and to evolve
    environments for inner populations to evolve within.
    """
    score = inner_log.filter(
        # Look only at live individuals in the last generation.
        (pl.col('id') > 0) &
        (pl.col('Generation') == INNER_GENERATIONS - 1)
    # Find the average hiff score in every location of the environment.
    ).group_by(
        'x', 'y'
    ).agg(
        pl.col('hiff').mean()
    # Use the median of those scores as a metric for how well the inner
    # population was able to evolve in these conditions. This rewards
    # populations that produce higher concentrations of high hiff scores in
    # larger areas of the environment.
    )['hiff'].median()

    # Handle missing values gracefully.
    return 0.0 if score is None else float(score)


def evolve(inner_population, environment, migration, crossover):
    """Evolve inner_population in environment, with the given configuration.
    """
    inner_population.randomize()
    for inner_generation in range(INNER_GENERATIONS):
        inner_population.evaluate(environment, inner_generation)

        if inner_generation + 1 < INNER_GENERATIONS:
            inner_population.propagate(
                environment, inner_generation, migration, crossover)

    # Collect the full evolutionary history in a data frame and return it.
    inner_log =  pl.DataFrame({
        field_name: data.flatten()
        for field_name, data in inner_population.to_numpy().items()
    }).hstack(
        index
    )
    return inner_log, outer_fitness(inner_log)


