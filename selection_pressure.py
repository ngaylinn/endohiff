"""Supplemental experiment visualizing the effect of selection pressure.
"""

import polars as pl
import taichi as ti

import constants
from constants import OUTPUT_PATH, INNER_GENERATIONS

# Override the global tournament size to see what happens when it's much bigger
# than our default setting (2). Note that we override this value *before*
# importing other modules that depend on it, so they see the updated value.
constants.TOURNAMENT_SIZE = 10

from environment import make_baym, make_flat
from chart import chart_hiff_dist
from chart_across_experiments import chart_hiff_comparison, compare_hiff_distributions
from evolve import evolve
from inner_population import InnerPopulation
from render import save_hiff_map

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)


def selection_pressure():
    variants = {
        'baym': make_baym,
        'flat': make_flat,
    }
    logs = {}
    for name, make_env in variants.items():
        path = OUTPUT_PATH / 'selection_pressure' / name
        path.mkdir(exist_ok=True, parents=True)

        # Actually run the experiment and capture the logs.
        env = make_env()
        inner_population = InnerPopulation()
        inner_log, _ = evolve(inner_population, env, True, True)
        logs[name] = inner_log
        last_gen = inner_log.filter(
            pl.col('Generation') == INNER_GENERATIONS - 1
        )

        # Generate visualizations for each experiment.
        save_hiff_map(path, name, last_gen)
        chart_hiff_dist(path, name, inner_log)

    # Merge the logs from the baym and stretched conditions so we can compare
    # them head-to-head.
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

    # Generate charts and calculate statistical significance comparing results
    # across these conditions.
    chart_hiff_comparison(
        head_to_head, 'environment', 'selection_pressure',
        OUTPUT_PATH / 'selection_pressure'/ f'hiff_dist.png')
    with open(OUTPUT_PATH / 'selection_pressure' / 'mannwitneyu.txt', 'w') as file:
        compare_hiff_distributions(head_to_head, 'environment', file)


if __name__ == '__main__':
    selection_pressure()
