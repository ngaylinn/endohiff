"""Simulate evolution of a symbiont population in a spatial environment.
"""

from functools import cache

import numpy as np
import polars as pl

from constants import INNER_GENERATIONS


@cache
def get_index(e, g, w, h, c):
    """Get a table of metadata with the same shape as the raw simulation data.

    By hstack'ing the index with the raw inner log data, we correctly annotate
    which environment, generation, and position of each individual. The
    arguments to this function describe the shape of the experiment data:
        e: number of environments
        g: number of generations
        w: width of all environments
        h: height of all environments
        c: carrying capacity for each location in all environments
    """
    # Compute an index enumerating the generation number and position of each
    # individual in a population over evolutionary time. This will be combined
    # with raw data from the simulator to produce log files.
    return pl.DataFrame({
        'env': np.arange(e).repeat(g * w * h * c),
        'Generation': np.tile(np.arange(g).repeat(w * h * c), e),
        'x': np.tile(np.arange(w).repeat(h * c), e * g),
        'y': np.tile(np.arange(h).repeat(c), e * g * w),
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
    index = get_index(*inner_population.pop.shape)
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


