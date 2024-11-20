from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange
from scipy.stats import bootstrap

from constants import INNER_GENERATIONS, OUTPUT_PATH, MAX_HIFF
from environment import ENVIRONMENTS

NUM_RUNS = 10 #bringing this down for memory problems

def plot_mean_and_bootstrapped_ci_over_time(experiment_results, name, x_label="Generation", y_label="Fitness", y_limit=None):
    fig, ax = plt.subplots()  # generate figure and axes

    this_input_data = experiment_results[name]
    total_generations = this_input_data.shape[1]

    bootstrap_ci = np.zeros((2, total_generations))
    for this_gen in range(total_generations):
        res = bootstrap((this_input_data[:, this_gen],), np.mean, confidence_level=0.95, n_resamples=100)
        bootstrap_ci[:, this_gen] = res.confidence_interval

    ax.plot(np.arange(total_generations), np.mean(this_input_data, axis=0), label=name)  # plot the fitness over time
    ax.fill_between(np.arange(total_generations), bootstrap_ci[0, :], bootstrap_ci[1, :], alpha=0.3)  # plot and fill the confidence interval for fitness over time
    ax.set_xlabel(x_label)  # add axes labels
    ax.set_ylabel(y_label)
    if y_limit:
        ax.set_ylim(y_limit[0], y_limit[1])
    plt.legend(loc='best')  # add legend

    plt.show()

def chart_across_experiments():
    experiment_results = np.load(OUTPUT_PATH / 'experiment_results.npy', allow_pickle=True).item()

    frames = []
    for env in ENVIRONMENTS.keys():
        for crossover in [True, False]:
            for migration in [True, False]:
                path = OUTPUT_PATH / f'migration_{migration}_crossover_{crossover}' / f'{env}'
                for run_num in range(NUM_RUNS):
                    log_file = path / f'inner_log_run_{run_num}.parquet'
                    if log_file.exists():
                        frames.append(pl.read_parquet(log_file).with_columns(
                            environment=pl.lit(env),
                            variant=pl.lit(f'migration_{migration}_crossover_{crossover}'),
                            crossover=pl.lit(crossover),
                            migration=pl.lit(migration)
                        ))

    all_data = pl.concat(frames)

    num_artifacts = 4 * 2 * len(ENVIRONMENTS)
    progress = trange(num_artifacts)

    # Iterate through all the unique combinations to plot
    for crossover in [True, False]:
        for migration in [True, False]:
            for env in ENVIRONMENTS.keys():
                variant_path = OUTPUT_PATH / f'migration_{migration}_crossover_{crossover}'
                variant_path.mkdir(parents=True, exist_ok=True)

                name = f'migration_{migration}_crossover_{crossover}'
                condition_key = f'{env}_migration_{migration}_crossover_{crossover}'

                plot_mean_and_bootstrapped_ci_over_time(experiment_results, condition_key, x_label="Generation", y_label="Fitness")

                variant_data = all_data.filter(
                    # Looking only at living individuals...
                    (pl.col('id') > 0) &
                    # From this algorithm variant...
                    (pl.col('variant') == name) &
                    # In the last generation only...
                    (pl.col('Generation') == INNER_GENERATIONS - 1)
                ).group_by(
                    'environment', 'Generation', 'x', 'y', maintain_order=True
                ).agg(
                    pl.col('hiff').mean().alias('Mean Hiff')
                )

                if len(variant_data) > 0:
                    fig = sns.displot(
                        variant_data.to_pandas(), x='Mean Hiff', kind='kde',
                        hue='environment', aspect=1.33)
                    plt.xlim(200, MAX_HIFF)
                    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.8, 0.8))
                    fig.set(title=f'Population HIFF Distribution ({name})')
                    fig.savefig(variant_path / f'hiff_dist.png', dpi=600)
                    plt.tight_layout()
                    plt.close()
                progress.update()

    # Plot across environments
    for env in ENVIRONMENTS.keys():
        env_data = all_data.filter(
            (pl.col('id') > 0) &
            (pl.col('environment') == env) &
            (pl.col('Generation') == INNER_GENERATIONS - 1)
        ).group_by(
            'variant', 'Generation', 'x', 'y', maintain_order=True
        ).agg(
            pl.col('hiff').mean().alias('Mean Hiff')
        )

        if len(env_data) > 0:
            fig = sns.displot(
                env_data.to_pandas(), x='Mean Hiff', kind='kde',
                hue='variant', aspect=1.33)
            plt.xlim(200, MAX_HIFF)
            sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.60, 0.8))
            fig.set(title=f'Population HIFF Distribution ({env})')
            fig.savefig(OUTPUT_PATH / f'hiff_dist_{env}.png', dpi=600)
            plt.tight_layout()
            plt.close()
        progress.update()

if __name__ == '__main__':
    chart_across_experiments()
