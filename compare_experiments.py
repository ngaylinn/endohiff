"""Compare hiff score distributions between two experiment conditions.
"""

from argparse import ArgumentParser
from itertools import combinations
import sys

import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import mannwhitneyu
import seaborn as sns
from tqdm import trange

from constants import INNER_GENERATIONS, OUTPUT_PATH, MAX_HIFF, MIN_HIFF
from run_experiment import get_all_trials


def summarize_hiff_scores(inner_log):
    return inner_log.filter(
        (pl.col('id') > 0) &
        (pl.col('Generation') == INNER_GENERATIONS - 1)
    # Average hiff scores for each location in the
    # environment, so we can analyze relative concentrations
    ).group_by(
        'x', 'y'
    ).agg(
        pl.col('hiff').mean().alias('Mean Hiff')
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

    hiff_score_frames = []
    for variant_name, env_name, trial, log_file in all_trials:
        inner_log = pl.read_parquet(log_file)
        hiff_score_frames.append(
            summarize_hiff_scores(
                inner_log
            ).with_columns(
                environment=pl.lit(env_name),
                variant=pl.lit(variant_name),
                trial=pl.lit(trial)
            ))
        tick_progress()
    return pl.concat(hiff_score_frames)


def get_aggregate_logs(verbose):
    hiff_scores_path = OUTPUT_PATH / 'hiff_scores.parquet'
    if hiff_scores_path.exists():
        if verbose > 0:
            print()
            print('Reusing aggregated log data.')
        hiff_scores = pl.read_parquet(hiff_scores_path)
    else:
        hiff_scores = aggregate_logs(verbose)
        hiff_scores.write_parquet(hiff_scores_path)
    return hiff_scores


def compare_hiff_distributions(data, column, file=None):
    """Use Mann Witney U test to compare result distributions.

    This function compares the local mean hiff score distributions between
    experiments whose data are distinguished by the value in column.
    """
    conditions = data[column].unique()
    max_len = max(map(len, conditions))
    # Do pairwise comparisons of all the different conditions found in data.
    for (cond_a, cond_b) in combinations(conditions, 2):
        # Isolate and compare data across conditions.
        data_a = data.filter(pl.col(column) == cond_a)['Mean Hiff']
        data_b = data.filter(pl.col(column) == cond_b)['Mean Hiff']
        _, p_less = mannwhitneyu(data_a, data_b, alternative='less')
        _, p_greater = mannwhitneyu(data_a, data_b, alternative='greater')

        # Print a summary (optionally to the given file)
        if p_less > p_greater:
            template = f'{{0:>{max_len}}} > {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_greater}', file=file)
        else:
            template = f'{{0:>{max_len}}} < {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_less}', file=file)


def chart_hiff_comparison(data, column, file):
    """Chart a comparison of mean hiff distributions.

    This function compares the local mean hiff score distributions between
    experiments whose data are distinguished by the value in column, saving the
    results to file, using name in the title.
    """
    fig = sns.displot(
        data, x='Mean Hiff', kind='kde', hue=column, aspect=1.33)
    plt.xlim(MIN_HIFF, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.6, 0.8))
    fig.set(title=f'Population HIFF Distribution')
    fig.savefig(file, dpi=600)
    plt.close()


def compare_experiments(path, data, column, suffix=''):
    """Compare hiff scores between different experimental variants.

    The results from all experiments to compare should be in data, with the
    value in column used to identify the conditions to compare. All outputs are
    saved to path, optionally annotated with the given filename suffix.
    """
    with open(path / f'mannwhitneyu{suffix}.txt', 'w') as file:
        compare_hiff_distributions(data, column, file)
    chart_hiff_comparison(
        data, column, path / f'hiff_dist{suffix}.png')


def main(verbose):
    """Visualize comparisons between all the primary experiments.
    """
    # Read the final hiff score data from all experiments that have been run.
    all_data = get_aggregate_logs(verbose)
    variants = all_data['variant'].unique()
    environments = all_data['environment'].unique()

    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        print()
        print('Charting comparisons...')
        num_artifacts = len(variants) + len(environments)
        tick_progress = trange(num_artifacts).update
    else:
        tick_progress = lambda: None

    # For each variant (ie, w/ and w/o crossover, migration), compare
    # performance of that variant across all environments.
    for variant_name in variants:
        variant_path = OUTPUT_PATH / variant_name
        variant_path.mkdir(parents=True, exist_ok=True)

        variant_data = all_data.filter(pl.col('variant') == variant_name)
        compare_experiments(variant_path, variant_data, 'environment')
        tick_progress()

    # For each environment (ie, random, flat, baym), compare performance of all
    # variants in that environment.
    for env_name in environments:
        env_data = all_data.filter(pl.col('environment') == env_name)
        compare_experiments(OUTPUT_PATH, env_data, 'variant', f'_{env_name}')
        tick_progress()


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate charts to compare results across experiment.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
