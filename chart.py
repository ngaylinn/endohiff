from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange
from scipy.stats import bootstrap

from constants import INNER_GENERATIONS, MAX_HIFF, MAX_POPULATION_SIZE, MIN_HIFF, OUTPUT_PATH
# from run_experiments import CONDITION_NAMES
from environment import ENVIRONMENTS

NUM_RUNS = 20


# Making ridgeplots with Seaborn generates lots of these warnings.
warnings.filterwarnings('ignore', category=UserWarning, message='Tight layout not applied')

def plot_mean_and_bootstrapped_ci_over_time(df, value_col, x_label="Generation", y_label="Value", y_limit=None, title=""):
    fig, ax = plt.subplots()  # generate figure and axes
    generations = df['Generation'].unique().to_list()
    mean_values = df.group_by('Generation').agg(pl.col(value_col).mean())
    
    bootstrap_ci = np.zeros((2, len(generations)))
    for i, gen in enumerate(generations):
        res = bootstrap((df.filter(pl.col('Generation') == gen)[value_col].to_numpy(),), np.mean, confidence_level=0.95, n_resamples=1000)
        bootstrap_ci[:, i] = res.confidence_interval
    
    ax.plot(generations, mean_values[value_col], label=value_col)  # plot the mean value over time
    ax.fill_between(generations, bootstrap_ci[0, :], bootstrap_ci[1, :], alpha=0.3)  # plot and fill the confidence interval for the value over time
    ax.set_xlabel(x_label)  # add axes labels
    ax.set_ylabel(y_label)
    if y_limit:
        ax.set_ylim(y_limit[0], y_limit[1])
    plt.legend(loc='best')  # add legend
    # plt.title(title)
    # plt.savefig(title.replace(" ", "_") + ".png", dpi=600)  # Save the figure
    return fig

def chart_fitness(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('fitness').max())
    fig = sns.relplot(data=expt_data, x='generation', y='fitness', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max fitness score ({name})')
    fig.savefig(path / 'fitness.png', dpi=600)
    plt.close()


def chart_hiff_max(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('hiff').max())
    fig = sns.relplot(data=expt_data, x='generation', y='hiff', kind='line')
    plt.ylim(MIN_HIFF, MAX_HIFF)
    fig.set(title=f'Max HIFF score ({name})')
    fig.savefig(path / 'hiff_max.png', dpi=600)
    plt.close()


def chart_hiff_sum(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg(pl.col('hiff').sum())
    overall_sum = expt_data['hiff'].sum()
    fig = sns.relplot(data=expt_data, x='generation', y='hiff', kind='line')
    plt.ylim(0, MAX_HIFF * MAX_POPULATION_SIZE)
    fig.set(title=f'Population HIFF Sum ({name}, total={overall_sum})')
    fig.savefig(path / 'hiff_sum.png', dpi=600)
    plt.close()


def chart_hiff_dist(path, name, expt_data):
    expt_data = expt_data.filter(
        # Looking only at living individuals...
        (pl.col('id') > 0) &
        # Sample every ten generations...
        (pl.col('Generation') % 10 == 9)
    ).group_by(
        # For all cells across all generations...
        'Generation', 'x', 'y'
    ).agg(
        # Find the mean hiff score for all individuals in this cell.
        pl.col('hiff').mean().alias('Mean Hiff')
    )

    sns.set_theme(style='white', rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    grid = sns.FacetGrid(expt_data, row='Generation', hue='Generation',
                         aspect=15, height=0.5, palette=pal)
    plt.xlim(0, MAX_HIFF)
    grid.map(sns.kdeplot, 'Mean Hiff', bw_adjust=1.5,
             clip_on=False, fill=True, alpha=1.0)
    grid.map(sns.kdeplot, 'Mean Hiff', bw_adjust=1.5,
             clip_on=False, color='w')

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, f'Gen {label}', ha='left', va='center',
                transform=ax.transAxes)
    grid.map(label, 'Mean Hiff')

    grid.refline(y=0, linestyle='-', clip_on=False)
    grid.figure.subplots_adjust(hspace=-0.25)
    grid.set_titles('')
    grid.set(yticks=[], ylabel='')
    grid.despine(bottom=True, left=True)
    grid.figure.suptitle(f'Hiff score distribution ({name})')
    grid.figure.supylabel('Density')
    grid.figure.savefig(path / 'hiff_dist.png', dpi=600)
    plt.close()
    # Restore the default colormap so we don't alter other charts generated
    # after this one.
    plt.set_cmap('viridis')


def chart_population_size(path, name, expt_data):
    expt_data = expt_data.group_by('generation').agg((pl.col('id') > 0).sum())
    fig = sns.relplot(data=expt_data, x='generation', y='id', kind='line')
    plt.ylim(0, MAX_POPULATION_SIZE)
    fig.set(title=f'Population Size ({name})')
    fig.savefig(path / 'pop_size.png', dpi=600)
    plt.close()


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


def chart_fitness_diversity(path, name, df):
    fig = plot_mean_and_bootstrapped_ci_over_time(df, 'Fitness Diversity', x_label='Generation', y_label='Fitness Diversity', y_limit=(0, 50), title=f'Fitness Diversity ({name})')

    # fig = sns.relplot(data=df, x='Generation', y='Fitness Diversity', kind='line')
    plt.ylim=(0, 50)  # Set y-axis limits from 0 to 50 - consistent for all environments
    fig.set(title=f'Fitness Diversity ({name})')
    fig.savefig(path / 'fitness_diversity.png', dpi=600)
    plt.close()

def chart_genetic_diversity(path, name, df):
    fig = plot_mean_and_bootstrapped_ci_over_time(df, 'Genetic Diversity', x_label='Generation', y_label='Genetic Diversity', y_limit=(0, 50), title=f'Genetic Diversity ({name})')
    # fig = sns.relplot(data=df, x='Generation', y='Genetic Diversity', kind='line')
    plt.ylim=(0, 50)  # Set y-axis limits from 0 to 50 - consistent for all environments
    fig.set(title=f'Genetic Diversity ({name})')
    fig.savefig(path / 'genetic_diversity.png', dpi=600)
    plt.close()

def chart_all_results():
    from run_experiments import CONDITION_NAMES

    # TODO: Consider using styles from Aquarel project?
    sns.set_theme()

    num_artifacts = 4 * 3 * len(CONDITION_NAMES)
    progress = trange(num_artifacts)
    # Iterate through all the unique combinations to plot
    for crossover in [True, False]:
        for migration in [True, False]:
            for env in ENVIRONMENTS.keys():
                name = env
                path = OUTPUT_PATH / f'migration_{migration}_crossover_{crossover}' / f'{env}'

                frames = []
                whole_pop_metrics_frames = []

                for run_num in range(NUM_RUNS):

                    try:
                        inner_log_path = path / f'inner_log_run_{run_num}.parquet'
                        whole_pop_metrics_path = path / f'whole_pop_metrics_run_{run_num}.parquet'

                        if inner_log_path.exists():
                            frames.append(pl.read_parquet(inner_log_path))
                        if whole_pop_metrics_path.exists():
                            whole_pop_metrics_frames.append(pl.read_parquet(whole_pop_metrics_path))
                    except Exception as e:
                        print(f"Could not process {inner_log_path} or {whole_pop_metrics_path}: {e}")

                if frames:
                    expt_data = pl.concat(frames)
                    chart_hiff_dist(path, name, expt_data)
                    progress.update()

                if whole_pop_metrics_frames:
                    whole_pop_metrics = pl.concat(whole_pop_metrics_frames)

                    chart_fitness_diversity(path, name, whole_pop_metrics)
                    progress.update()

                    chart_genetic_diversity(path, name, whole_pop_metrics)
                    progress.update()           


                # chart_fitness(path, name, expt_data)
                # progress.update()

                # chart_hiff_max(path, name, expt_data)
                # progress.update()

                # chart_hiff_sum(path, name, expt_data)
                # progress.update()



                # chart_population_size(path, name, expt_data)
                # progress.update()

                # chart_survival(path, name, expt_data)
                # progress.update()



if __name__ == '__main__':
    chart_all_results()
