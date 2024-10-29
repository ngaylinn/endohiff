import numpy as np
import polars as pl
import taichi as ti
import pandas as pd

from constants import MAX_HIFF, OUTPUT_PATH
from environment import (
    make_flat_environment, make_random_environment, make_designed_environment)
from evolve import evolve
from inner_population import InnerPopulation

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)

CONDITIONS = {
    'control_flat': make_flat_environment,
    'control_random': make_random_environment,
    'control_designed': make_designed_environment,
}

CONDITION_NAMES = list(CONDITIONS.keys())


def print_summary(name, expt_data):
    best_hiff, best_bitstr = expt_data.filter(
        pl.col('hiff') == pl.col('hiff').max()
    ).select(
        'hiff', 'bitstr'
    )
    print(f'Experiment condition: {name}')
    print(f'{len(best_hiff)} individual(s) found the highest score '
          f'({best_hiff[0]} out of a possible {MAX_HIFF})')
    print(f'Example: {best_bitstr[0]:064b}')
    print()

# def compute_population_diversity(filename: str) -> float:
#     # added 10/29 (Anna)
#     # takes in inner_population.log_population parquet file and calculates diversity from that
#     # NOTE: not sure if this is the most efficient way to do this 
#     df = pd.read_parquet(filename)
#     bitstrings = df.to_numpy()
#     allele_frequencies = np.mean(bitstrings, axis=0)
#     diversity = np.std(allele_frequencies)
#     return diversity

# def compute_population_diversity(inner_population) -> pl.DataFrame:
#     # added 10/29 (Anna)
#     # NOTE: not sure if this is the most efficient way to do this
#     # Compute diversity based on the inner population
#     bitstrings = inner_population.to_numpy().get('bitstr')  # Extract the bitstrings
#     allele_frequencies = np.mean(bitstrings, axis=0)
#     diversity = np.std(allele_frequencies)

#     # Create a DataFrame to store the results
#     diversity_log = pl.DataFrame({
#         'generation': range(inner_population.generations),
#         'diversity': [diversity] * inner_population.generations  # Repeat the diversity value
#     })

#     return diversity_log


def run_experiments():
    # Make a place to save results.
    OUTPUT_PATH.mkdir(exist_ok=True)

    # Run each experiment condition in turn.
    for name, make_environment in CONDITIONS.items():
        # Make a place to put experiment results.
        path = OUTPUT_PATH / name
        path.mkdir(exist_ok=True)

        # Run the condition and cache the results.
        inner_population = InnerPopulation()
        environment = make_environment()
        inner_log = evolve(inner_population, environment)

        # Save the experiment logs to disk.
        inner_log.write_parquet(path / 'inner_log.parquet')

        # Save the final environment for this experiment. Currently this is
        # always static, but eventually it will be evolved.
        np.savez(path / 'env.npz', **environment.to_numpy())

        # # Log diversity history as a DataFrame
        # diversity_df = pl.DataFrame(inner_population.diversity_history, columns=['generation', 'diversity', 'average_fitness'])
        # diversity_df.write_parquet(path / 'diversity_inner_log.parquet')
        
        # Summarize results on the command line.
        print_summary(name, inner_log)


if __name__ == '__main__':
    run_experiments()
