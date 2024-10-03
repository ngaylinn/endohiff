import numpy as np
import polars as pl
import taichi as ti
from tqdm import trange

from constants import CARRYING_CAPACITY, ENVIRONMENT_SHAPE, MAX_SCORE, NUM_GENERATIONS
from environment import make_flat_environment
from inner_population import InnerPopulation

ti.init(ti.cuda)


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


def evolve():
    # Setup
    pop = InnerPopulation()
    pop.randomize(0)
    environment = make_flat_environment()

    # Random search (TODO: make this evolutionary!)
    progress = trange(NUM_GENERATIONS)
    for generation in progress:
        best_hiff = pop.evaluate(environment, generation)
        progress.set_description(f'F={best_hiff}')

        if generation + 1 < NUM_GENERATIONS:
            pop.propagate(generation)

    # Record the full history and save to disk.
    history = pl.DataFrame({
        field_name: data.flatten()
        for field_name, data in pop.to_numpy().items()
    }).hstack(
        index
    )
    history.write_parquet('output/history.parquet')

    # Summarize results.
    best_hiff, best_bitstr = history.filter(
        pl.col('hiff') == pl.col('hiff').max()
    ).select(
        'hiff', 'bitstr'
    )
    print(f'{len(best_hiff)} individual(s) found the highest score '
          f'({best_hiff[0]} out of a possible {MAX_SCORE})')
    print(f'Example: {best_bitstr[0]:064b}')


if __name__ == '__main__':
    evolve()
