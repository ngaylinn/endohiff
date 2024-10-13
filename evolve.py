import numpy as np
import polars as pl
from tqdm import trange

from constants import CARRYING_CAPACITY, ENVIRONMENT_SHAPE, INNER_GENERATIONS
from inner_population import InnerPopulation


# Compute an index enumerating the generation number and position of each
# individual in a population over evolutionary time.
g = INNER_GENERATIONS
w, h = ENVIRONMENT_SHAPE
c = CARRYING_CAPACITY
index = pl.DataFrame({
    'generation': np.arange(g).repeat(w * h * c),
    'x': np.tile(np.arange(w).repeat(h * c), g),
    'y': np.tile(np.arange(h).repeat(c), g * w),
})


def evolve(environment):
    # Setup
    population = InnerPopulation()
    population.randomize()

    # Random search (TODO: make this evolutionary!)
    for inner_generation in range(INNER_GENERATIONS):
        population.evaluate(environment, inner_generation)

        if inner_generation + 1 < INNER_GENERATIONS:
            population.propagate(environment, inner_generation)

    # Collect the full evolutionary history in a data frame and return it.
    return pl.DataFrame({
        field_name: data.flatten()
        for field_name, data in population.to_numpy().items()
    }).hstack(
        index
    )
