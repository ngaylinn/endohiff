"""Supplemental experiment comparing variations of the Baym environment.
"""

import numpy as np
import polars as pl
import taichi as ti

from constants import ENVIRONMENT_SHAPE, INNER_GENERATIONS, OUTPUT_PATH
from chart import chart_hiff_dist
from chart_across_experiments import chart_hiff_comparison, compare_hiff_distributions
from environment import Environment, make_baym
from evolve import evolve
from inner_population import InnerPopulation
from render import save_env_map, save_hiff_map

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)


def make_stretched():
    # Ramp up and down like Baym, but make the "step" that got the highest
    # fitness scores much, much wider.
    ramp_up_and_down = np.array([
         64,  64, 128, 128, 192, 192, 256, 256,
        320, 320, 320, 320, 320, 320, 320, 320,
        320, 320, 320, 320, 320, 320, 320, 320,
        320, 320, 320, 320, 320, 384, 384, 448,
        448, 384, 384, 320, 320, 320, 320, 320,
        320, 320, 320, 320, 320, 320, 320, 320,
        320, 320, 320, 320, 320, 320, 320, 320,
        256, 256, 192, 192, 128, 128,  64,  64])

    env = Environment()
    env.min_fitness.from_numpy(
        np.broadcast_to(
            np.expand_dims(ramp_up_and_down, 1),
            ENVIRONMENT_SHAPE))
    env.weights.fill(1.0)
    return env


# This is just a variation on the flat environment, but using the minimum
# fitness threshold that got the best fitness score in a garduated setting.
def make_noramp():
    env = Environment()
    env.min_fitness.from_numpy(np.full(ENVIRONMENT_SHAPE, 320))
    env.weights.fill(1.0)
    return env


def baym_variants():
    variants = {
        'baym': make_baym,
        'stretched': make_stretched,
        'no_ramp': make_noramp,
    }
    logs = {}

    for name, make_env in variants.items():
        path = OUTPUT_PATH / 'baym_variants' / name
        path.mkdir(exist_ok=True, parents=True)

        # Actually run the experiment and capture the logs.
        env = make_env()
        inner_population = InnerPopulation()
        inner_log, _ = evolve(inner_population, env, True, True)
        last_gen = inner_log.filter(
            pl.col('Generation') == INNER_GENERATIONS - 1
        )
        logs[name] = inner_log

        # Generate visualizations for each experiment.
        save_env_map(path, env.to_numpy())
        save_hiff_map(path, last_gen)
        try:
            chart_hiff_dist(path, inner_log)
        except ValueError:
            # If the population dies out immediately, rendering a distribution
            # chart fails, but that's okay. Just skip that visualization.
            pass

    # Merge the logs from the baym and stretched conditions so we can compare
    # them head-to-head.
    head_to_head = pl.concat((
        logs['baym'].with_columns(environment=pl.lit('baym')),
        logs['stretched'].with_columns(environment=pl.lit('stretched'))
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
        head_to_head, 'environment', 'baym variants',
        OUTPUT_PATH / 'baym_variants'/ f'hiff_dist.png')
    with open(OUTPUT_PATH / 'baym_variants' / 'mannwhitneyu.txt', 'w') as file:
        compare_hiff_distributions(head_to_head, 'environment', file)


if __name__ == '__main__':
    baym_variants()
