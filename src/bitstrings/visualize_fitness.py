from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from ..constants import ENV_NAMES, MAX_HIFF, MIN_HIFF
from ..graphics import ENV_NAME_COLORS


def load_all_log_files(path):
    all_frames = []
    for env_name in ENV_NAMES:
        for trial_path in (path / env_name).glob('trial_?/'):
            inner_log = pl.read_parquet(
                trial_path / 'inner_log.parquet'
            ).filter(
                pl.col('alive') == True
            ).select(
                'fitness', 'Generation'
            ).with_columns(
                Environment=pl.lit(env_name),
                Trial=pl.lit(trial_path.stem)
            )
            all_frames.append(inner_log)

    all_data = pl.concat(all_frames)
    del all_frames
    return all_data


def indicate_best_trials(path, all_data):
    for env_name in ENV_NAMES:
        best_trial = all_data.filter(
            pl.col('Environment') == env_name
        ).filter(
            pl.col('fitness') == pl.col('fitness').max()
        ).filter(
            pl.col('Generation') == pl.col('Generation').min()
        )['Trial'][0]
        with open(path / env_name / 'best_trial.txt', 'w') as file:
            print(best_trial, file=file, flush=True)


def summarize(all_data):
    return pl.concat((
        all_data.group_by(
            'Environment', 'Trial', 'Generation', maintain_order=True
        ).agg(
            pl.col('fitness').max().alias('Fitness')
        ).with_columns(
            Aggregation=pl.lit('max')
        ),
        all_data.group_by(
            'Environment', 'Trial', 'Generation', maintain_order=True
        ).agg(
            pl.col('fitness').mean().alias('Fitness')
        ).with_columns(
            Aggregation=pl.lit('mean')
        )))


def full_range(vector):
    return (vector.min(), vector.max())


def chart_all(path, all_data):
    sns.relplot(
        all_data, x='Generation', y='Fitness', kind='line',
        hue='Environment', hue_order=ENV_NAMES, style='Aggregation',
        errorbar=full_range, height=2.667, aspect=2, legend=False)
    plt.ylim((MIN_HIFF, MAX_HIFF))
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(path / 'fitness_comparison.png', dpi=150)
    plt.close()


def chart_one(path, all_data, env_name):
    env_data = all_data.filter( pl.col('Environment') == env_name)
    sns.relplot(
        env_data, x='Generation', y='Fitness', kind='line',
        color=ENV_NAME_COLORS[env_name], style='Aggregation',
        errorbar=full_range, height=3, legend=False)
    plt.ylim((MIN_HIFF, MAX_HIFF))
    #plt.xlabel('')
    #plt.ylabel('')
    plt.tight_layout()
    plt.savefig(path / env_name / 'bitstring_fitness.png', dpi=150)
    plt.close()


def main(path):
    all_data = load_all_log_files(path)
    indicate_best_trials(path, all_data)

    all_data = summarize(all_data)
    chart_all(path, all_data)
    for env_name in ENV_NAMES:
        chart_one(path, all_data, env_name)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Chart bit string evolution for all environments.')
    parser.add_argument(
        'path', type=Path,
        help='A directory containing logs from all trials in all environments')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
