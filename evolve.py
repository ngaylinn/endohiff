import numpy as np
import polars as pl
from tqdm import trange

from constants import CARRYING_CAPACITY, ENVIRONMENT_SHAPE, NUM_GENERATIONS
from inner_population import InnerPopulation


# Compute an index enumerating the generation number and position of each
# individual in a population over evolutionary time.
g = NUM_GENERATIONS
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
    population.randomize(0)

    # Random search (TODO: make this evolutionary!)
    progress = trange(NUM_GENERATIONS)
    for generation in progress:
        best_hiff = population.evaluate(environment, generation)
        progress.set_description(f'F={best_hiff}')

        if generation + 1 < NUM_GENERATIONS:
            population.propagate(generation)

    # Record the full history and save to disk.
    history = pl.DataFrame({
        field_name: data.flatten()
        for field_name, data in population.to_numpy().items()
    }).hstack(
        index
    )
    return history
