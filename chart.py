from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import MAX_HIFF, MIN_HIFF

def chart_fitness(name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('fitness').max())
    fig = sns.relplot(data=expt_data, x='generation', y='fitness', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max fitness score ({name})')
    fig.savefig(f'output/{name}/fitness.png', dpi=600)


def chart_hiff(name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('hiff').max())
    fig = sns.relplot(data=expt_data, x='generation', y='hiff', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max HIFF score ({name})')
    fig.savefig(f'output/{name}/hiff.png', dpi=600)


def chart_all_results():
    # TODO: Consider using styles from Aquarel project?
    sns.set_theme()

    history = pl.read_parquet('output/history.parquet')
    conditions = history.select('condition').unique().to_series().to_list()

    num_artifacts = 2 * len(conditions)
    progress = trange(num_artifacts)
    for name in conditions:
        Path(f'output/{name}').mkdir(exist_ok=True)
        expt_data = history.filter(pl.col('condition') == name)

        chart_fitness(name, expt_data)
        progress.update()

        chart_hiff(name, expt_data)
        progress.update()


if __name__ == '__main__':
    chart_all_results()
