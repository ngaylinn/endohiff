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

# init. whole-pop metric df
whole_pop_metrics_df = pl.DataFrame({
    'generation': np.arange(INNER_GENERATIONS),
    'fitness_diversity': [0.0] * INNER_GENERATIONS,  # Initialize with zeros
    'genetic_diversity': [0.0] * INNER_GENERATIONS
    # TODO: add new metrics later :)
})


def evolve(inner_population, environment, migration, crossover):
    global whole_pop_metrics_df

    inner_population.randomize()
    fitness_diversity_list = []
    genetic_diversity_list = []
    bitstrings_list = []  # List to store bitstrings of all individuals
    for inner_generation in range(INNER_GENERATIONS):
        inner_population.evaluate(environment, inner_generation)
        # TODO: add same stuff for other whole-population metrics
        current_gen_fitness_diversity = inner_population.fitness_diversity.to_numpy()
        current_gen_genetic_diversity = inner_population.genetic_diversity.to_numpy()

        fitness_diversity_list.append(current_gen_fitness_diversity)
        genetic_diversity_list.append(current_gen_genetic_diversity)


        # Update the fitness value for this generation using set
        whole_pop_metrics_df = whole_pop_metrics_df.with_columns(
            pl.when(pl.col('generation') == inner_generation)
            .then(current_gen_fitness_diversity)
            .otherwise(pl.col('fitness_diversity'))
            .alias('fitness_diversity')
        ).with_columns(
            pl.when(pl.col('generation') == inner_generation)
            .then(current_gen_genetic_diversity)
            .otherwise(pl.col('genetic_diversity'))
            .alias('genetic_diversity')
        )

        if inner_generation + 1 < INNER_GENERATIONS:
            inner_population.propagate(environment, inner_generation, migration, crossover)

    # Collect the full evolutionary history in a data frame and return it.
    inner_log =  pl.DataFrame({
        field_name: data.flatten()
        for field_name, data in inner_population.to_numpy().items()
    }).hstack(
        index
    )
    return inner_log, whole_pop_metrics_df


