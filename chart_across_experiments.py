from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import mannwhitneyu
import seaborn as sns
from tqdm import trange

from constants import OUTPUT_PATH, MAX_HIFF
from environment import ENVIRONMENTS


def compare_hiff_distributions(file, data, column):
    conditions = data[column].unique()
    max_len = max(map(len, conditions))
    for (cond_a, cond_b) in combinations(conditions, 2):
        data_a = data.filter(pl.col(column) == cond_a)['Mean Hiff']
        data_b = data.filter(pl.col(column) == cond_b)['Mean Hiff']
        _, p_less = mannwhitneyu(data_a, data_b, alternative='less')
        _, p_greater = mannwhitneyu(data_a, data_b, alternative='greater')
        if p_less > p_greater:
            template = f'{{0:>{max_len}}} > {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_greater}', file=file)
        else:
            template = f'{{0:>{max_len}}} < {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_less}', file=file)


def chart_across_experiments():
    # Read the final hiff score data from all the experiments.
    all_data = pl.read_parquet('output/*/*/hiff_scores.parquet')

    variants = all_data['variant'].unique()
    environments = all_data['environment'].unique()

    num_artifacts = len(variants) + len(environments)
    progress = trange(num_artifacts)

    for variant_name in variants:
        variant_path = OUTPUT_PATH / variant_name
        variant_path.mkdir(parents=True, exist_ok=True)

        hiff_scores = all_data.filter(
            pl.col('variant') == variant_name
        ).select(
            'environment', 'Mean Hiff'
        )
        with open(variant_path / 'mannwhitneyu.txt', 'w') as file:
            compare_hiff_distributions(file, hiff_scores, 'environment')

        if len(hiff_scores) > 0:
            fig = sns.displot(
                hiff_scores, x='Mean Hiff', kind='kde',
                hue='environment', aspect=1.33)
            plt.xlim(200, MAX_HIFF)
            sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.8, 0.8))
            fig.set(title=f'Population HIFF Distribution ({variant_name})')
            fig.savefig(variant_path / f'hiff_dist.png', dpi=600)
            plt.close()
        progress.update()

    for env_name in environments:
        hiff_scores = all_data.filter(
            pl.col('environment') == env_name
        ).select(
            'variant', 'Mean Hiff'
        )
        with open(OUTPUT_PATH / f'mannwhitneyu_{env_name}.txt', 'w') as file:
            compare_hiff_distributions(file, hiff_scores, 'variant')

        if len(hiff_scores) > 0:
            fig = sns.displot(
                hiff_scores, x='Mean Hiff', kind='kde',
                hue='variant', aspect=1.33)
            plt.xlim(200, MAX_HIFF)
            sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.60, 0.8))
            fig.set(title=f'Population HIFF Distribution ({env_name})')
            fig.savefig(OUTPUT_PATH / f'hiff_dist_{env_name}.png', dpi=600)
            plt.close()
        progress.update()

if __name__ == '__main__':
    chart_across_experiments()
