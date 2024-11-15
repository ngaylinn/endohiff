from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import INNER_GENERATIONS, OUTPUT_PATH, MAX_HIFF
from environment import ENVIRONMENTS

def chart_across_experiments():
    frames = []

    for crossover in [True, False]:
        for migration in [True, False]:
            for env in ENVIRONMENTS.keys():
                path = OUTPUT_PATH / f'migration_{migration}_crossover_{crossover}' / f'{env}'
                if (path / 'inner_log.parquet').exists():
                    frames.append(pl.read_parquet(
                        path / 'inner_log.parquet'
                    ).with_columns(
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

                variant_data = all_data.filter(
                    # Looking only at living individuals...
                    (pl.col('id') > 0) &
                    # From this algorithm variant...
                    (pl.col('variant') == name) &
                    # In the last generation only...
                    (pl.col('Generation') == INNER_GENERATIONS - 1)
                ).group_by(
                    # For all cells across all generations...
                    'environment', 'Generation', 'x', 'y', maintain_order=True
                ).agg(
                    # Find the mean hiff score for individuals in this cell.
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

    for env in ENVIRONMENTS.keys():
        env_data = all_data.filter(
            # Looking only at living individuals...
            (pl.col('id') > 0) &
            # From this environment...
            (pl.col('environment') == env) &
            # In the last generation only...
            (pl.col('Generation') == INNER_GENERATIONS - 1)
        ).group_by(
            # For all cells across all generations...
            'variant', 'Generation', 'x', 'y', maintain_order=True
        ).agg(
            # Find the mean hiff score for individuals in this cell.
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
