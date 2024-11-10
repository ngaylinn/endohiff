import numpy as np
import polars as pl
import taichi as ti
import pandas as pd

from constants import MAX_HIFF, OUTPUT_PATH
from environment import ENVIRONMENTS
from evolve import evolve
from inner_population import InnerPopulation

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)

# TODO: Once we add an evolved environment, include it here, also.
CONDITION_NAMES = ENVIRONMENTS.keys()

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


def run_experiments():

    # Make a place to save results.
    OUTPUT_PATH.mkdir(exist_ok=True)

    # Run each experiment condition in turn.
    # TODO: Once we add an evolved environment, include it here, also.
    for name, make_environment in ENVIRONMENTS.items():
        # Make a place to put experiment results.
        path = OUTPUT_PATH / name
        path.mkdir(exist_ok=True)

        # Run the condition and cache the results.
        inner_population = InnerPopulation()
        environment = make_environment()
        inner_log, whole_pop_metrics = evolve(inner_population, environment)

        # Save the experiment logs to disk.
        inner_log.write_parquet(path / 'inner_log.parquet')

        # Save the whole_pop_metrics DataFrame if you want to keep it
        whole_pop_metrics.write_parquet(path / 'whole_pop_metrics.parquet')

        # Save the final environment for this experiment. Currently this is
        # always static, but eventually it will be evolved.
        np.savez(path / 'env.npz', **environment.to_numpy())

        
        # Summarize results on the command line.
        print_summary(name, inner_log)


if __name__ == '__main__':
    run_experiments()
