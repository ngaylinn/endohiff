from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import MAX_HIFF, MAX_POPULATION_SIZE, MIN_HIFF, OUTPUT_PATH
from run_experiments import CONDITION_NAMES

def chart_fitness(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('fitness').max())
    fig = sns.relplot(data=expt_data, x='generation', y='fitness', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max fitness score ({name})')
    fig.savefig(path / 'fitness.png', dpi=600)


def chart_hiff_max(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('hiff').max())
    fig = sns.relplot(data=expt_data, x='generation', y='hiff', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max HIFF score ({name})')
    fig.savefig(path / 'hiff_max.png', dpi=600)


def chart_hiff_sum(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('hiff').sum())
    overall_sum = expt_data['hiff'].sum()
    fig = sns.relplot(data=expt_data, x='generation', y='hiff', kind='line')
    plt.ylim(0, MAX_HIFF * MAX_POPULATION_SIZE)
    fig.set(title=f'Population HIFF Sum ({name}, total={overall_sum})')
    fig.savefig(path / 'hiff_sum.png', dpi=600)


def chart_population_size(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg((pl.col('id') > 0).sum())
    fig = sns.relplot(data=expt_data, x='generation', y='id', kind='line')
    plt.ylim(0, MAX_POPULATION_SIZE)
    fig.set(title=f'Population Size ({name})')
    fig.savefig(path / 'pop_size.png', dpi=600)

def chart_diversity(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('diversity').max())  # Adjusted to max diversity per generation if needed
    fig = sns.relplot(data=expt_data, x='generation', y='diversity', kind='line')
    plt.ylim(0, 1)  # Set limits according to expected diversity range (adjust as needed)
    fig.set(title=f'Population Diversity ({name})')
    fig.savefig(path / 'diversity.png', dpi=600)


def chart_all_results():
    # TODO: Consider using styles from Aquarel project?
    # TODO: add diversity chart
    sns.set_theme()

    num_artifacts = 4 * len(CONDITION_NAMES)
    progress = trange(num_artifacts)
    for name in CONDITION_NAMES:
        path = OUTPUT_PATH / name
        expt_data = pl.read_parquet(path / 'inner_log.parquet')

        chart_fitness(path, name, expt_data)
        progress.update()

        chart_hiff_max(path, name, expt_data)
        progress.update()

        chart_hiff_sum(path, name, expt_data)
        progress.update()

        chart_population_size(path, name, expt_data)
        progress.update()

        try:
            chart_diversity(path, name, expt_data)
            progress.update()
        except:
            print("diversity didn't save")


if __name__ == '__main__':
    chart_all_results()
