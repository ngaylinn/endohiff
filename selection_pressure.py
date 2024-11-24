import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from chart_across_experiments import compare_hiff_distributions

import constants
from constants import OUTPUT_PATH, INNER_GENERATIONS, MAX_HIFF

constants.TOURNAMENT_SIZE = 10

from environment import make_baym, make_flat
from chart import chart_hiff_dist
from evolve import evolve
from inner_population import InnerPopulation
from render import save_hiff_map


def selection_pressure():
    variants = {
        'baym': make_baym,
        'flat': make_flat,
    }
    logs = {}
    for name, make_env in variants.items():
        path = OUTPUT_PATH / 'selection_pressure' / name
        path.mkdir(exist_ok=True, parents=True)

        env = make_env()
        inner_population = InnerPopulation()
        inner_log, _, _ = evolve(inner_population, env, True, True)
        logs[name] = inner_log
        last_gen = inner_log.filter(
            pl.col('Generation') == INNER_GENERATIONS - 1
        )

        save_hiff_map(path, name, last_gen)
        chart_hiff_dist(path, name, inner_log)

    head_to_head = pl.concat((
        logs['baym'].with_columns(environment=pl.lit('baym')),
        logs['flat'].with_columns(environment=pl.lit('flat'))
    )).filter(
        (pl.col('id') > 0) &
        (pl.col('Generation') == INNER_GENERATIONS - 1)
    ).group_by(
        'environment', 'x', 'y'
    ).agg(
        pl.col('hiff').mean().alias('Mean Hiff')
    )

    fig = sns.displot(
        head_to_head, x='Mean Hiff', kind='kde', hue='environment', aspect=1.33)
    plt.xlim(200, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.40, 0.8))
    fig.set(title=f'Population HIFF Distribution (baym variants)')
    fig.savefig(OUTPUT_PATH / 'selection_pressure'/ f'hiff_dist.png', dpi=600)
    plt.close()
    with open(OUTPUT_PATH / 'selection_pressure' / 'mannwitneyu.txt', 'w') as file:
        compare_hiff_distributions(file, head_to_head, 'environment')


if __name__ == '__main__':
    selection_pressure()
