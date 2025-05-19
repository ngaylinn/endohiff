"""Compare hiff score distributions between two experiment conditions.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from constants import MAX_HIFF, MIN_HIFF


HUE_ORDER = ['baym', 'flat', 'cppn', 'high', 'gap']
COLORS = {
    env_name: sns.color_palette('colorblind')[i]
    for i, env_name in enumerate(HUE_ORDER)
}

def compare_trials(path, env_name, env_data):
    sns.set_palette('colorblind')

    best_trial = env_data.filter(
        pl.col('fitness') == pl.col('fitness').max()
    ).filter(
        pl.col('Generation') == pl.col('Generation').min()
    )['Trial'][0]
    with open(path / env_name / 'best_trial.txt', 'w') as file:
        print(best_trial, file=file, flush=True)

    env_data = env_data.filter(
        pl.col('alive')
    ).group_by(
        'Trial', 'Generation', maintain_order=True
    ).agg(
        pl.col('fitness').max().alias('Max Fitness'),
        pl.col('fitness').mean().alias('Mean Fitness')
    )

    frames = []
    for (trial, generation, max_fitness, mean_fitness) in env_data.rows():
        frames.append(pl.DataFrame({
            'Generation': generation,
            'Fitness': max_fitness,
            'Aggregation': 'max'}))
        frames.append(pl.DataFrame({
            'Generation': generation,
            'Fitness': mean_fitness,
            'Aggregation': 'mean'}))
    data = pl.concat(frames)

    def full_range(vector):
        return (vector.min(), vector.max())

    sns.relplot(
        data, x='Generation', y='Fitness', kind='line',
        color=COLORS[env_name], style='Aggregation', errorbar=full_range,
        height=3.25, legend=False
        )
    plt.ylim((MIN_HIFF, MAX_HIFF))
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(path / env_name / 'env_fitness.png', dpi=300)
    plt.close()


def main(path):
    """Visualize comparisons between all the primary experiments.
    """
    all_frames = []
    env_names = ['flat', 'baym', 'cppn']
    for env_name in env_names:
        for trial_path in (path / env_name).glob('*/'):
            inner_log = pl.read_parquet(
                trial_path / 'inner_log.parquet'
            ).drop(
                'bitstr', 'one_count',
            ).with_columns(
                Environment=pl.lit(env_name),
                Trial=pl.lit(trial_path.stem)
            )
            all_frames.append(inner_log)

    all_data = pl.concat(all_frames)
    del all_frames

    # TODO: Add back code to generate combined fitness chart for all
    # environments.
    for env_name in env_names:
        env_data = all_data.filter( pl.col('Environment') == env_name)
        compare_trials(path, env_name, env_data)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate charts to compare results across experiment.')
    parser.add_argument(
        'path', type=Path, help='Path to results from multiple experiments')
    args = vars(parser.parse_args())

    sys.exit(main(**args))

