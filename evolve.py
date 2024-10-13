import numpy as np
import polars as pl
from tqdm import trange

from constants import (
    ALL_ENVIRONMENTS_SHAPE, CARRYING_CAPACITY, ENVIRONMENT_SHAPE,
    INNER_GENERATIONS, NUM_HOST_ENVIRONMENTS, OUTER_GENERATIONS)
from inner_population import InnerPopulation
from outer_population import OuterPopulation

# Compute an index enumerating the generation number and position of each
# individual in a population over evolutionary time.
e = NUM_HOST_ENVIRONMENTS
g = INNER_GENERATIONS
w, h = ENVIRONMENT_SHAPE
c = CARRYING_CAPACITY
index = pl.DataFrame({
    'host_environment': np.arange(e).repeat(g * w * h * c),
    'inner_generation': np.tile(np.arange(g).repeat(w * h * c), e),
    'x': np.tile(np.arange(w).repeat(h * c), e * g),
    'y': np.tile(np.arange(h).repeat(c), e * g * w),
})


def summarize(inner_population, inner_log):
    # Compute aggregate metrics.
    diversity = np.zeros(
        (INNER_GENERATIONS, ) + ALL_ENVIRONMENTS_SHAPE)
    inner_population.compute_diversity(diversity)

    # Aggregate logged data across the full population of each environment
    # location.
    return inner_log.group_by(
        'generation', 'x', 'y'
    ).agg(
        # TODO: Compute and store other statistics, like CIs?
        pl.col('fitness').mean(),
        pl.col('hiff').max(),
    ).with_columns(
        # Instead of keeping all the bit strings, just keep aggregate metrics
        # about them.
        diversity=diversity
    )


# TODO: Run multiple trials for statistical significance?
def evolve(environment):
    # Setup the inner population.
    inner_population = InnerPopulation()
    inner_log = None

    # If a static environment was not provided, create an outer population that
    # will generate an evolved environment.
    outer_population = None
    outer_log = None
    if environment is None:
        outer_population = OuterPopulation()
        outer_population.randomize()
        outer_log = pl.DataFrame()

    # Random search (TODO: make this evolutionary!)
    progress = trange(OUTER_GENERATIONS)
    for outer_generation in progress:
        if outer_population:
            environment = outer_population.make_environment()

        # Evolve the inner population on the GPU
        inner_population.randomize()
        for inner_generation in range(INNER_GENERATIONS):
            inner_population.evaluate(environment, inner_generation)

            if inner_generation + 1 < INNER_GENERATIONS:
                inner_population.propagate(environment, inner_generation)

        # Grab the log from the inner loop for processing.
        inner_log = pl.DataFrame({
            field_name: data.flatten()
            for field_name, data in inner_population.to_numpy().items()
        }).hstack(
            index
        ).with_columns(
            outer_generation=outer_generation
        )

        # Give a status update.
        best_hiff = inner_log.select('hiff').max().item()
        progress.set_description(f'F={best_hiff}')

        # If we're evolving an outer population, update logs and propage.
        if outer_population:
            # TODO: Add additional metrics tracking the outer population?
            summary = summarize(inner_population, inner_log)
            outer_log = pl.concat([outer_log, summary])
            if outer_generation + 1 < OUTER_GENERATIONS:
                # TODO: Make this evolutionary!
                outer_population.propagate()

    # Return the full outer log and the inner log for the last generation.
    return inner_log, outer_log
