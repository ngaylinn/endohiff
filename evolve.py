import numpy as np
import polars as pl

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
    'diversity': np.arange(g).repeat(w * h * c),
})


def evolve(inner_population, environment):
    inner_population.randomize()
    diversity_list = []
    for inner_generation in range(INNER_GENERATIONS):
        inner_population.evaluate(environment, inner_generation)

        diversity_list.append(inner_population.pop_diversity.to_numpy())

        #print(f"diversity at {inner_generation} = {inner_population.pop_diversity[inner_generation]}")

        if inner_generation + 1 < INNER_GENERATIONS:
            inner_population.propagate(environment, inner_generation)


    # Collect the full evolutionary history in a data frame and return it.
    return pl.DataFrame({
        field_name: data.flatten()
        for field_name, data in inner_population.to_numpy().items()
    }).hstack(
        index
    )
