from pathlib import Path

import numpy as np
import polars as pl
import taichi as ti

from constants import MAX_HIFF
from environment import (
    make_flat_environment, make_random_environment, make_designed_environment)
from evolve import evolve

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)

conditions = {
    'control_flat': make_flat_environment,
    'control_random': make_random_environment,
    'control_designed': make_designed_environment,
}


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
    path = Path('output')
    path.mkdir(exist_ok=True)

    # Run each experiment condition in turn.
    frames = []
    for name, make_environment in conditions.items():
        # Run the condition and cache the results.
        environment = make_environment()
        expt_data = evolve(environment).with_columns(
            condition=pl.lit(name)
        )
        frames.append(expt_data)

        # Save the final environment for this experiment. Currently this is
        # always static, but eventually it will be evolved.
        (path / name).mkdir(exist_ok=True)
        np.savez(path / name / 'env.npz', **environment.to_numpy())

        # Summarize results on the command line.
        print_summary(name, expt_data)

    # Save all experiment results to disk.
    history = pl.concat(frames)
    history.write_parquet('output/history.parquet')


if __name__ == '__main__':
    run_experiments()
