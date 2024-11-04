from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import INNER_GENERATIONS, MAX_HIFF, MAX_POPULATION_SIZE, MIN_HIFF, OUTPUT_PATH
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


def chart_hiff_density(path, name, expt_data):
    # For all cells across all generations...
    expt_data = expt_data.group_by(
        'generation', 'x', 'y'
    ).agg(
        # Sum all the hiff scores in this cell and divide by the number of
        # living individuals in this cell.
        (pl.col('hiff').sum() / (pl.col('id') > 0).len()).alias('mean_hiff')
    )
    # Plot average hiff score for every cell across generations, using a
    # scatter plot so we can see the distribution of cells with high or low
    # hiff scores over evolutionary time.
    fig = sns.relplot(data=expt_data, x='generation', y='mean_hiff',
                      kind='scatter', alpha=0.1, marker='.')
    plt.ylim(0, MAX_HIFF)
    fig.set(title=f'Hiff density ({name})')
    fig.savefig(path / 'hiff_density.png', dpi=600)
    plt.close()


def chart_population_size(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg((pl.col('id') > 0).sum())
    fig = sns.relplot(data=expt_data, x='generation', y='id', kind='line')
    plt.ylim(0, MAX_POPULATION_SIZE)
    fig.set(title=f'Population Size ({name})')
    fig.savefig(path / 'pop_size.png', dpi=600)


def chart_survival(path, name, expt_data):
    # TODO: This doesn't look at mates from crossover, so in some sense it is
    # undercounting which individuals contribute to the next generation.
    parents = expt_data.filter(pl.col('parent') > 0)['parent'].unique()

    parent_data = expt_data.filter(
        # Restrict to just individuals who ARE parents, since this cuts the amount
        # of data down substantially.
        pl.col('id').is_in(parents)
    ).select(
        # Rename the id field to parent so we can line up the data for each
        # parent with their children using a join operation.
        pl.col('id').alias('parent'),
        # Rename all the fields of interest to indicate they come from the
        # parent.
        pl.col('fitness').alias('parent_fitness'),
        pl.col('x').alias('parent_x'),
        pl.col('y').alias('parent_y'),
    )

    # Pull in data from parents and associate it with each individual.
    expt_data = expt_data.join(
        parent_data, on='parent', how='left', coalesce=False,
    ).with_columns(
        has_children = pl.col('id').is_in(parents),
        more_fit = pl.col('fitness') > pl.col('parent_fitness'),
        migrant = (
            (pl.col('x') != pl.col('parent_x')) &
            (pl.col('y') != pl.col('parent_y')))
    ).filter(
        pl.col('generation') < (INNER_GENERATIONS - 2)
    )

    # Compute survival rates for migrants and for individuals more fit than
    # their parents.
    survival = pl.concat([
        expt_data.filter(
            pl.col('migrant')
        ).group_by(
            'generation', maintain_order=True
        ).agg(
            (pl.col('has_children').sum() / pl.len()).alias('survival_rate')
        ).with_columns(
            category=pl.lit('migrant')
        ),
        expt_data.filter(
            pl.col('more_fit')
        ).group_by(
            'generation', maintain_order=True
        ).agg(
            (pl.col('has_children').sum() / pl.len()).alias('survival_rate')
        ).with_columns(
            category=pl.lit('more_fit')
        )
    ])

    fig = sns.relplot(data=survival, x='generation', y='survival_rate',
                      kind='line', hue='category')
    plt.ylim(0.0, 1.0)
    fig.set(title=f'Survival rate ({name})')
    fig.savefig(path / 'survival.png', dpi=600)
    plt.close()


def chart_diversity(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('diversity').max())  # Adjusted to max diversity per generation if needed
    fig = sns.relplot(data=expt_data, x='generation', y='diversity', kind='line')
    plt.ylim(0, 1)  # Set limits according to expected diversity range (adjust as needed)
    fig.set(title=f'Population Diversity ({name})')
    fig.savefig(path / 'diversity.png', dpi=600)


def chart_all_results():
    # TODO: Consider using styles from Aquarel project?
    # TODO: fix diversity chart
    sns.set_theme()

    num_artifacts = 7 * len(CONDITION_NAMES)
    progress = trange(num_artifacts)
    for name in CONDITION_NAMES:
        path = OUTPUT_PATH / name
        expt_data = pl.read_parquet(path / 'inner_log.parquet')

        #chart_fitness(path, name, expt_data)
        progress.update()

        #chart_hiff_max(path, name, expt_data)
        progress.update()

        #chart_hiff_sum(path, name, expt_data)
        progress.update()

        chart_hiff_density(path, name, expt_data)
        progress.update()

        #chart_population_size(path, name, expt_data)
        progress.update()

        #chart_survival(path, name, expt_data)
        progress.update()

        try:
            chart_diversity(path, name, expt_data)
            progress.update()
        except:
            print("diversity didn't save")


if __name__ == '__main__':
    chart_all_results()
