"""Visualize the performance of an outer population over one and many trials.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import NUM_TRIALS
from visualize_inner_population import save_env_map


def chart_outer_fitness(data, path, hue=None):
    sns.relplot(data=data, x='Generation', y='Fitness', kind='line', hue=hue)
    plt.savefig(path / f'outer_fitness.png', dpi=300)


def main(path, verbose):
    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        num_artifacts = NUM_TRIALS + 1
        tick_progress = trange(num_artifacts).update
    else:
        tick_progress = lambda: None

    for trial in range(NUM_TRIALS):
        env_data = np.load(path / f'cppn_{trial}.npy')
        save_env_map(path, env_data, f'cppn_{trial}')
        tick_progress()

    all_data = pl.read_parquet(path / 'outer_log.parquet')
    chart_outer_fitness(all_data, path, hue='Trial')
    tick_progress()

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate visualizations of outer population performanc3e.')
    parser.add_argument(
        'path', type=Path, help='Where to find experiment result data.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
