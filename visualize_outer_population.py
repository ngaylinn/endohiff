"""Visualize the performance of an outer population over one and many trials.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import NUM_TRIALS


def chart_outer_fitness(data, path, hue=None, suffix=''):
    sns.relplot(data=data, x='Generation', y='Fitness', kind='line', hue=hue)
    plt.savefig(path / f'outer_fitness{suffix}.png', dpi=600)


def main(path, verbose):
    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        num_artifacts = NUM_TRIALS + 2
        tick_progress = trange(num_artifacts).update
    else:
        tick_progress = lambda: None

    all_data = pl.read_parquet(path / 'outer_log.parquet')

    # Load and chart data for each trial separately.
    for t in range(NUM_TRIALS):
        trial_path = path / f'trial{t}'
        trial_data = all_data.filter(pl.col('Trial') == t)
        chart_outer_fitness(trial_data, trial_path)
        tick_progress()

    # Chart data from all trials, both aggregated and broken down by trial.
    chart_outer_fitness(all_data, path)
    tick_progress()
    chart_outer_fitness(all_data, path, hue='Trial', suffix='_by_trial')
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
