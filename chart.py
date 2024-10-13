from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import MAX_HIFF, MIN_HIFF, OUTPUT_PATH
from run_experiments import CONDITION_NAMES

def chart_fitness(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('fitness').max())
    fig = sns.relplot(data=expt_data, x='generation', y='fitness', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max fitness score ({name})')
    fig.savefig(path / 'fitness.png', dpi=600)


def chart_hiff(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('hiff').max())
    fig = sns.relplot(data=expt_data, x='generation', y='hiff', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max HIFF score ({name})')
    fig.savefig(path / 'hiff.png', dpi=600)


def chart_all_results():
    # TODO: Consider using styles from Aquarel project?
    sns.set_theme()

    num_artifacts = 2 * len(CONDITION_NAMES)
    progress = trange(num_artifacts)
    for name in CONDITION_NAMES:
        path = OUTPUT_PATH / name
        expt_data = pl.read_parquet(path / 'inner_log.parquet')

        chart_fitness(path, name, expt_data)
        progress.update()

        chart_hiff(path, name, expt_data)
        progress.update()


if __name__ == '__main__':
    chart_all_results()
