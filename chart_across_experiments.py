"""Tools for comparing performance between two experiment conditions.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import mannwhitneyu
import seaborn as sns
from tqdm import trange

from constants import OUTPUT_PATH, MAX_HIFF


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


def chart_hiff_comparison(data, column, name, path):
    """Chart a comparison of mean hiff distributions.

    This function compares the local mean hiff score distributions between
    experiments whose data are distinguished by the value in column, saving the
    results to path, using name in the title.
    """
    fig = sns.displot(
        data, x='Mean Hiff', kind='kde', hue=column, aspect=1.33)
    plt.xlim(200, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.6, 0.8))
    fig.set(title=f'Population HIFF Distribution ({name})')
    fig.savefig(path, dpi=600)
    plt.close()


def chart_across_experiments():
    """Visualize comparisons between all the primary experiments.
    """
    # Read the final hiff score data from all the experiments.
    all_data = pl.read_parquet('output/*/*/hiff_scores.parquet')

    variants = all_data['variant'].unique()
    environments = all_data['environment'].unique()

    num_artifacts = len(variants) + len(environments)
    progress = trange(num_artifacts)

    # For each variant (ie, w/ and w/o crossover, migration), compare
    # performance of that variant across all environments.
    for variant_name in variants:
        variant_path = OUTPUT_PATH / variant_name
        variant_path.mkdir(parents=True, exist_ok=True)

        # Get just the data for this variant.
        hiff_scores = all_data.filter(
            pl.col('variant') == variant_name
        ).select(
            'environment', 'Mean Hiff'
        )

        # Chart a comparison between these distributions and compute
        # significance.
        with open(variant_path / 'mannwhitneyu.txt', 'w') as file:
            compare_hiff_distributions(hiff_scores, 'environment', file)
        chart_hiff_comparison(
            hiff_scores, 'environment', variant_name,
            variant_path / f'hiff_dist.png')
        progress.update()

    # For each environment (ie, random, flat, baym), compare performance of all
    # variants in that environment.
    for env_name in environments:
        # Get just the data for this variant.
        hiff_scores = all_data.filter(
            pl.col('environment') == env_name
        ).select(
            'variant', 'Mean Hiff'
        )

        # Chart a comparison between these distributions and compute
        # significance.
        with open(OUTPUT_PATH / f'mannwhitneyu_{env_name}.txt', 'w') as file:
            compare_hiff_distributions(hiff_scores, 'variant', file)
        if len(hiff_scores) > 0:
            chart_hiff_comparison(
                hiff_scores, 'variant', env_name,
                OUTPUT_PATH / f'hiff_dist_{env_name}.png')
        progress.update()

if __name__ == '__main__':
    chart_across_experiments()
