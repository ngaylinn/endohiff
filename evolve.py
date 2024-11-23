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
    'Generation': np.arange(g).repeat(w * h * c),
    'x': np.tile(np.arange(w).repeat(h * c), g),
    'y': np.tile(np.arange(h).repeat(c), g * w),
})

# init. whole-pop metric df
whole_pop_metrics_df = pl.DataFrame({
    'Generation': np.arange(INNER_GENERATIONS),
    'Fitness Diversity': [0.0] * INNER_GENERATIONS,  # Initialize with zeros
    'Genetic Diversity': [0.0] * INNER_GENERATIONS
    # TODO: add new metrics later :)
})


def outer_fitness(inner_log):
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
    return 0.0 if score is None else float(score)


def evolve(inner_population, environment, migration, crossover):
    global whole_pop_metrics_df

    inner_population.randomize()
    fitness_diversity_list = []
    genetic_diversity_list = []
    for inner_generation in range(INNER_GENERATIONS):
        inner_population.evaluate(environment, inner_generation)
        # TODO: add same stuff for other whole-population metrics
        current_gen_fitness_diversity = inner_population.fitness_diversity.to_numpy()
        current_gen_genetic_diversity = inner_population.genetic_diversity.to_numpy()

        fitness_diversity_list.append(current_gen_fitness_diversity)
        genetic_diversity_list.append(current_gen_genetic_diversity)


        # Update the fitness value for this generation using set
        whole_pop_metrics_df = whole_pop_metrics_df.with_columns(
            pl.when(pl.col('Generation') == inner_generation)
            .then(current_gen_fitness_diversity)
            .otherwise(pl.col('Fitness Diversity'))
            .alias('Fitness Diversity')
        ).with_columns(
            pl.when(pl.col('Generation') == inner_generation)
            .then(current_gen_genetic_diversity)
            .otherwise(pl.col('Genetic Diversity'))
            .alias('Genetic Diversity')
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
    return inner_log, whole_pop_metrics_df, outer_fitness(inner_log)


