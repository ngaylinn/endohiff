"""Compare hiff score distributions between two experiment conditions.
"""

from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from scipy import stats
import seaborn as sns
from tqdm import trange

from constants import INNER_GENERATIONS, NUM_TRIALS, OUTPUT_PATH, MAX_HIFF, MIN_HIFF
from run_experiment import get_all_trials


HUE_ORDER = ['baym', 'flat', 'cppn', 'high', 'gap']
COLORS = {
    env_name: sns.color_palette('colorblind')[i]
    for i, env_name in enumerate(HUE_ORDER)
}


# Making ridgeplots with Seaborn generates lots of these warnings.
warnings.filterwarnings('ignore', category=UserWarning, message='Tight layout not applied')


def summarize_fitness(inner_log):
    return inner_log.filter(
        pl.col('alive') & (pl.col('Generation') == INNER_GENERATIONS - 1)
    # Average fitness scores for each location for each environment, so we can
        # analyze relative concentrations
    ).group_by(
        'Environment', 'x', 'y', maintain_order=True
    ).agg(
        pl.col('fitness').mean().cast(pl.Float64).alias('Mean Fitness')
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

    frames = []
    for variant_name, env_name, trial, log_file in all_trials:
        inner_log = pl.read_parquet(log_file)
        frames.append(
            summarize_fitness(
                inner_log
            ).with_columns(
                environment=pl.lit(env_name),
                variant=pl.lit(variant_name),
                trial=pl.lit(trial)
            ))
        tick_progress()
    return pl.concat(frames)


def get_aggregate_logs(verbose):
    aggregate_fitness_path = OUTPUT_PATH / 'fitness.parquet'
    if aggregate_fitness_path.exists():
        if verbose > 0:
            print()
            print('Reusing aggregated log data.')
        fitness = pl.read_parquet(aggregate_fitness_path)
    else:
        fitness = aggregate_logs(verbose)
        fitness.write_parquet(aggregate_fitness_path)
    return fitness


def compare_fitness_distributions(data, file=None):
    """Use Mann Witney U test to compare result distributions.

    This function compares the local mean fitness score distributions between
    experiments whose data are distinguished by the value in column.
    """
    conditions = data['Environment'].unique()
    max_len = max(map(len, conditions))
    # Do pairwise comparisons of all the different conditions found in data.
    for (cond_a, cond_b) in combinations(conditions, 2):
        # Isolate and compare data across conditions.
        data_a = data.filter(pl.col('Environment') == cond_a)['Mean Fitness']
        data_b = data.filter(pl.col('Environment') == cond_b)['Mean Fitness']
        _, p_less = stats.mannwhitneyu(data_a, data_b, alternative='less')
        _, p_greater = stats.mannwhitneyu(data_a, data_b, alternative='greater')

        # Print a summary (optionally to the given file)
        if p_less > p_greater:
            template = f'{{0:>{max_len}}} > {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_greater}', file=file)
        else:
            template = f'{{0:>{max_len}}} < {{1:<{max_len}}}'
            print(template.format(cond_a, cond_b), f'p = {p_less}', file=file)


def chart_fitness_comparison(data, path):
    """Chart a comparison of mean fitness distributions.

    This function compares the local mean fitness score distributions between
    experiments whose data are distinguished by the value in column, saving the
    results to file, using name in the title.
    """
    sns.set_palette('colorblind')
    fig = sns.displot(
        data, x='Mean Fitness', kind='kde', hue='Environment',
        aspect=1.33, hue_order=HUE_ORDER, legend=False)
    plt.xlim(MIN_HIFF, MAX_HIFF)
#    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.7, 0.8))
    fig.savefig(path / f'final_fitness.png', dpi=300)
    plt.close()


def chart_fitness_dist(data, path):
    """Generate a ridgeplot showing fitness distribution over generations.
    """
    num_ridges = 10
    ridge_gap = INNER_GENERATIONS // num_ridges
    sample_generations = set(
        range(ridge_gap - 1, INNER_GENERATIONS, ridge_gap))
    data = data.filter(
        # Looking only at living individuals...
        pl.col('alive') &
        # Sample every ten generations...
        (pl.col('Generation').is_in(sample_generations))
    ).group_by(
        # For all cells across all generations...
        'Environment', 'Generation', 'x', 'y', maintain_order=True
    ).agg(
        # Find the mean fitness score for all individuals in this cell.
        pl.col('fitness').mean().alias('Local Mean Fitness')
    ).drop('x', 'y')

    # Since we only count living individuals, we might have no data for some
    # generations if the population went extinct. Make sure we can still
    # generate a chart by filling in data for missing generations.
    for c in data['Environment'].unique():
        c_data = data.filter(pl.col('Environment') == c)
        missing_generations = (
            sample_generations - set(c_data['Generation'].unique()))
        data = pl.concat((data, pl.DataFrame({
            'Environment': c,
            'Generation': sorted(missing_generations),
            'Local Mean Fitness': np.float32(0.0)
        })))

    # Add a column indicating the overall average fitness score so we can color
    # each ridge to match the performance at that time step.
    data = data.join(
        data.group_by(
            'Generation'
        ).agg(
            pl.col('Local Mean Fitness').mean().alias('color')
        ),
        on='Generation',
        how='inner',
        coalesce=True
    )

    # Set up the ridge plot visualization.
    sns.set_palette('colorblind')
    sns.set_theme(palette='colorblind', style='white',
                  rc={"axes.facecolor": (0, 0, 0, 0)})
    grid = sns.FacetGrid(
        data, row='Generation', hue='Environment', hue_order=HUE_ORDER,
        aspect=15, height=0.5, xlim=(0, MAX_HIFF))

    # Plot the mean fitness score for each sample generation and label it with the
    # generation number.
    def plot_ridge(x, color, label, data):
        ax = plt.gca()
        sns.kdeplot(x=data[x], fill=False, warn_singular=False, alpha=1.0,
                    color=color)
        # Only draw labels once (we happen to use the label 'baym' in all
        # the charts rendered by this function).
        if label == 'baym':
            generation = data['Generation'].unique().item()
            ax.text(0, 0.4, f'Gen {generation}', ha='left', va='center',
                    transform=ax.transAxes)

    grid.map_dataframe(plot_ridge, 'Local Mean Fitness')

    # Apply styling and save results.
    grid.refline(y=0, linestyle='-', clip_on=False, lw=0.5)
    grid.set_titles('')
    grid.set(yticks=[], ylabel='')
    grid.despine(bottom=True, left=True)
    plt.tight_layout()
    grid.figure.subplots_adjust(hspace=-0.3)
    grid.figure.savefig(path / 'fitness.png', dpi=300)
    plt.close()


def compare_experiments(path, data):
    """Compare fitness scores between different experimental variants.

    The results from all experiments to compare should be in data, with the
    value in column used to identify the conditions to compare. All outputs are
    saved to path, optionally annotated with the given filename suffix.
    """
    chart_fitness_dist(data, path)
    data = summarize_fitness(data)
    with open(path / f'mannwhitneyu.txt', 'w') as file:
        compare_fitness_distributions(data, file)
    chart_fitness_comparison(data, path)


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

    overall_max_fitness = env_data['Max Fitness'].max()
    time_to_max_fitness = env_data.filter(
        pl.col('Max Fitness') == overall_max_fitness
    )['Generation'].min()

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
    #plt.axvline(x=time_to_max_fitness, ls=':', c='k')
    #plt.annotate(
    #    f'{overall_max_fitness} @\ng={time_to_max_fitness}',
    #    (time_to_max_fitness, overall_max_fitness),
    #    xytext=(time_to_max_fitness - 60, overall_max_fitness - 40),# textcoords='figure fraction',
    #    arrowprops={'arrowstyle': '->'})
    plt.tight_layout()
    plt.savefig(path / env_name / 'env_fitness.png', dpi=300)
    plt.close()


def run_t_tests(all_data, file):
    df = all_data.filter(
        pl.col('alive')
    ).group_by(
        'Environment', 'Trial'
    ).agg(
        pl.col('fitness').max().alias('max_fitness'),
        pl.col('fitness').mean().alias('mean_fitness'),
    )
    p_thresh = 0.05
    crit_t = stats.t.ppf(1 - p_thresh/2, NUM_TRIALS * 2 - 2)
    env_names = list(df['Environment'].unique())
    for env1, env2 in combinations(env_names, 2):
        for metric in ['max_fitness', 'mean_fitness']:
            env1_data = df.filter(pl.col('Environment') == env1)[metric]
            env2_data = df.filter(pl.col('Environment') == env2)[metric]
            t_stat, p_value = stats.ttest_ind(env1_data, env2_data)
            print(f'{env1} vs. {env2}, {metric}', file=file)
            env1_summary = ', '.join([f'{val:0.1f}' for val in env1_data])
            env2_summary = ', '.join([f'{val:0.1f}' for val in env2_data])
            print(f'[{env1_summary}] vs. [{env2_summary}]', file=file)
            print(f'{abs(t_stat)} >? {crit_t}', file=file)
            print(('no ' if abs(t_stat) <= crit_t else '') +
                  'significant difference', file=file)
            print(f'{p_value} <? {p_thresh}', file=file)
            print(('no ' if p_value > p_thresh else '') +
                  'evidence to reject null hypthesis', file=file)
            print(file=file)


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

    for env_name in env_names:
        env_data = all_data.filter( pl.col('Environment') == env_name)
        compare_trials(path, env_name, env_data)
    # with open(path / 't_tests.txt', 'w') as file:
    #     run_t_tests(all_data, file)
    #compare_experiments(path, all_data)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate charts to compare results across experiment.')
    parser.add_argument(
        'path', type=Path, help='Path to results from multiple experiments')
    args = vars(parser.parse_args())

    sys.exit(main(**args))

