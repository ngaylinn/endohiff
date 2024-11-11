from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from constants import INNER_GENERATIONS, OUTPUT_PATH, MAX_HIFF
from environment import ENVIRONMENTS


algorithm_variants = {
    'c0_m0': ('without crossover, without migration', False, False),
    'c0_m1': ('without crossover, with migration', False, True),
    'c1_m0': ('with crossover, without migration', True, False),
    'c1_m1': ('with crossover, with migration', True, True)
}

frames = []
for env in ENVIRONMENTS.keys():
    for key, (name, crossover, migration) in algorithm_variants.items():
        frames.append(pl.read_parquet(
            OUTPUT_PATH / env / f'log_{key}.parquet'
        ).with_columns(
            environment=pl.lit(env),
            variant=pl.lit(name),
            crossover=pl.lit(crossover),
            migration=pl.lit(migration)
        ))
all_data = pl.concat(frames)


for key, (name, crossover, migration) in algorithm_variants.items():
    variant_data = all_data.filter(
        # Looking only at living individuals...
        (pl.col('id') > 0) &
        # From this algorithm variant...
        (pl.col('variant') == name) &
        # In the last generation only...
        (pl.col('generation') == INNER_GENERATIONS - 1)
    ).group_by(
        # For all cells across all generations...
        'environment', 'generation', 'x', 'y', maintain_order=True
    ).agg(
        # Find the mean hiff score for individuals in this cell.
        pl.col('hiff').mean().alias('Mean Hiff')
    )

    fig = sns.displot(
        variant_data, x='Mean Hiff', kind='kde',
        hue='environment', aspect=1.33)
    plt.xlim(200, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.8, 0.8))
    fig.set(title=f'Population HIFF Distribution ({name})')
    fig.savefig(OUTPUT_PATH / f'hiff_dist_{key}.png', dpi=600)
    plt.tight_layout()
    plt.close()

for env in ENVIRONMENTS.keys():
    env_data = all_data.filter(
        # Looking only at living individuals...
        (pl.col('id') > 0) &
        # From this environment...
        (pl.col('environment') == env) &
        # In the last generation only...
        (pl.col('generation') == INNER_GENERATIONS - 1)

    ).group_by(
        # For all cells across all generations...
        'variant', 'generation', 'x', 'y', maintain_order=True
    ).agg(
        # Find the mean hiff score for individuals in this cell.
        pl.col('hiff').mean().alias('Mean Hiff')
    )

    fig = sns.displot(
        env_data, x='Mean Hiff', kind='kde',
        hue='variant', aspect=1.33)
    plt.xlim(200, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.60, 0.8))
    fig.set(title=f'Population HIFF Distribution ({env})')
    fig.savefig(OUTPUT_PATH / f'hiff_dist_{env}.png', dpi=600)
    plt.tight_layout()
    plt.close()
