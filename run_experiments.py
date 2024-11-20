import numpy as np
import polars as pl
import taichi as ti
import pandas as pd
from pathlib import Path
import pickle

from constants import MAX_HIFF, OUTPUT_PATH
from environment import ENVIRONMENTS
from evolve import evolve
from inner_population import InnerPopulation



# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)

# TODO: Once we add an evolved environment, include it here, also.
CONDITION_NAMES = ENVIRONMENTS.keys()
NUM_RUNS = 20
INNER_GENERATIONS = 100

def print_summary(name, expt_data, migration, crossover):
    best_hiff, best_bitstr = expt_data.filter(
        pl.col('hiff') == pl.col('hiff').max()
    ).select(
        'hiff', 'bitstr'
    )
    print(f'Experiment condition: {name}')
    print(f'Migration: {migration}, Crossover: {crossover}')
    print(f'{len(best_hiff)} individual(s) found the highest score '
          f'({best_hiff[0]} out of a possible {MAX_HIFF})')
    print(f'Example: {best_bitstr[0]:064b}')
    print()


def run_experiments():
    OUTPUT_PATH.mkdir(exist_ok=True)
    experiment_results = {}

    for migration in [True, False]:
        for crossover in [True, False]:
            subfolder = f'migration_{migration}_crossover_{crossover}'
            subfolder_path = OUTPUT_PATH / subfolder
            subfolder_path.mkdir(exist_ok=True)

            for name, make_environment in ENVIRONMENTS.items():
                if name not in experiment_results:
                    experiment_results[name] = {}
                
                condition_key = f'migration_{migration}_crossover_{crossover}'

                # Initialize the condition_key as an empty dictionary if it doesn't exist
                if condition_key not in experiment_results[name]:
                    experiment_results[name][condition_key] = {}

                for run_num in range(NUM_RUNS):
                    path = subfolder_path / name
                    path.mkdir(exist_ok=True)

                    inner_population = InnerPopulation()
                    environment = make_environment()
                    inner_log, whole_pop_metrics = evolve(inner_population, environment, migration, crossover)

                    inner_log.write_parquet(path / f'inner_log_run_{run_num}.parquet')
                    whole_pop_metrics.write_parquet(path / f'whole_pop_metrics_run_{run_num}.parquet')

                    np.savez(path / 'env.npz', **environment.to_numpy())

                    print_summary(name, inner_log, migration, crossover)
                    # Store the fitness values for each generation
                    experiment_results[name][condition_key][run_num] = inner_log.to_pandas().to_dict()

    np.save(OUTPUT_PATH / 'experiment_results.npy', experiment_results)


if __name__ == '__main__':
    run_experiments()
