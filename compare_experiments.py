"""Compare hiff score distributions between two experiment conditions.
"""

from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations
import sys

import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import mannwhitneyu
import seaborn as sns
from tqdm import trange

from constants import INNER_GENERATIONS, OUTPUT_PATH, MAX_HIFF, MIN_HIFF
from run_experiment import get_all_trials


def summarize_fitness(inner_log):
    return inner_log.filter(
        pl.col('alive') & (pl.col('Generation') == INNER_GENERATIONS - 1)
    # Average fitness scores for each location in the environment, so we can
    # analyze relative concentrations
    ).group_by(
        'x', 'y'
    ).agg(
        pl.col('fitness').mean().cast(pl.Float64).alias('Mean Fitness')
    # Drop the location data. We just want to make a histogram over locations
    # in the environment, so we need samples from each location, but don't need
    # to remember where the samples came from.
    ).drop(
        'x', 'y'
    )


def aggregate_logs(verbose):
    all_trials = list(get_all_trials())
    # Maybe show a progress bar as we process files.
    if verbose > 0:
        print()
        print('Aggregating log files...')
        tick_progress = trange(len(all_trials)).update
    else:
        tick_progress = lambda: None

    frames = []
    for variant_name, env_name, trial, log_file in all_trials:
        inner_log = pl.read_parquet(log_file)
        frames.append(
            summarize_fitness(
                inner_log
            ).with_columns(
                environment=pl.lit(env_name),
                variant=pl.lit(variant_name),
                trial=pl.lit(trial)
            ))
        tick_progress()
    return pl.concat(frames)


def get_aggregate_logs(verbose):
    aggregate_fitness_path = OUTPUT_PATH / 'fitness.parquet'
    if aggregate_fitness_path.exists():
        if verbose > 0:
            print()
            print('Reusing aggregated log data.')
        fitness = pl.read_parquet(aggregate_fitness_path)
    else:
        fitness = aggregate_logs(verbose)
        fitness.write_parquet(aggregate_fitness_path)
    return fitness


def compare_fitness_distributions(data, column, file=None):
    """Use Mann Witney U test to compare result distributions.

    This function compares the local mean fitness score distributions between
    experiments whose data are distinguished by the value in column.
    """
    conditions = data[column].unique()
    max_len = max(map(len, conditions))
    # Do pairwise comparisons of all the different conditions found in data.
    for (cond_a, cond_b) in combinations(conditions, 2):
        # Isolate and compare data across conditions.
        data_a = data.filter(pl.col(column) == cond_a)['Mean Fitness']
        data_b = data.filter(pl.col(column) == cond_b)['Mean Fitness']
        _, p_less = mannwhitneyu(data_a, data_b, alternative='less')
        _, p_greater = mannwhitneyu(data_a, data_b, alternative='greater')

        # Print a summary (optionally to the given file)
        if p_less > p_greater:
            template = f'{{0:>{max_len}}} > {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_greater}', file=file)
        else:
            template = f'{{0:>{max_len}}} < {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_less}', file=file)


def chart_fitness_comparison(data, column, file):
    """Chart a comparison of mean fitness distributions.

    This function compares the local mean fitness score distributions between
    experiments whose data are distinguished by the value in column, saving the
    results to file, using name in the title.
    """
    fig = sns.displot(
        data, x='Mean Fitness', kind='kde', hue=column, aspect=1.33)
    plt.xlim(MIN_HIFF, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.7, 0.8))
    fig.set(title=f'Population Fitness Distribution')
    fig.savefig(file, dpi=600)
    plt.close()


def compare_experiments(path, data, column, suffix=''):
    """Compare fitness scores between different experimental variants.

    The results from all experiments to compare should be in data, with the
    value in column used to identify the conditions to compare. All outputs are
    saved to path, optionally annotated with the given filename suffix.
    """
    with open(path / f'mannwhitneyu{suffix}.txt', 'w') as file:
        compare_fitness_distributions(data, column, file)
    chart_fitness_comparison(
        data, column, path / f'fitness_dist{suffix}.png')


def main(path):
    """Visualize comparisons between all the primary experiments.
    """
    frames = []
    for env_path in path.glob('*/'):
        for trial_path in env_path.glob('*/'):
            inner_log = pl.read_parquet(trial_path / 'inner_log.parquet')
            frames.append(
                summarize_fitness(inner_log).with_columns(
                    Environment=pl.lit(env_path.stem)))
    all_data = pl.concat(frames)
    compare_experiments(path, all_data, 'Environment')

    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate charts to compare results across experiment.')
    parser.add_argument(
        'path', type=Path, help='Path to results from multiple experiments')
    args = vars(parser.parse_args())

    sys.exit(main(**args))

