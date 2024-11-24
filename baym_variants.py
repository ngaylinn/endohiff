import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from chart_across_experiments import compare_hiff_distributions

from constants import ENVIRONMENT_SHAPE, INNER_GENERATIONS, OUTPUT_PATH, MAX_HIFF
from chart import chart_hiff_dist
from environment import Environment, make_baym
from evolve import evolve
from inner_population import InnerPopulation
from render import save_env_map, save_hiff_map, calculate_genetic_diversity, render_genetic_diversity_map

def make_stretched():
    # Ramp up and down like Baym, but make the "step" that got the highest
    # fitness scores much, much wider.
    ramp_up_and_down = np.array([
         64,  64, 128, 128, 192, 192, 256, 256,
        320, 320, 384, 384, 384, 384, 384, 384,
        384, 384, 384, 384, 384, 384, 384, 384,
        384, 384, 384, 384, 384, 384, 384, 448,
        448, 384, 384, 384, 384, 384, 384, 384,
        384, 384, 384, 384, 384, 384, 384, 384,
        384, 384, 384, 384, 384, 384, 320, 320,
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
    env.min_fitness.from_numpy(np.full(ENVIRONMENT_SHAPE, 384))
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

        env = make_env()
        inner_population = InnerPopulation()
        inner_log, _, _ = evolve(inner_population, env, True, True)
        last_gen = inner_log.filter(
            pl.col('Generation') == INNER_GENERATIONS - 1
        )
        logs[name] = inner_log

        save_env_map(path, name, env.to_numpy())
        save_hiff_map(path, name, last_gen)
        genetic_diversity_map = calculate_genetic_diversity(
            inner_log, INNER_GENERATIONS - 1)
        render_genetic_diversity_map(path, name, genetic_diversity_map)
        try:
            chart_hiff_dist(path, name, inner_log)
        except ValueError:
            # If the population dies out immediately, rendering a distribution
            # chart fails, but that's okay. Just skip that visualization.
            pass

    # Compare the hiff distributions for the baym and stretched variants head
    # to head (we leave out the no_ramp variant, since the population dies out)
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

    fig = sns.displot(
        head_to_head, x='Mean Hiff', kind='kde', hue='environment', aspect=1.33)
    plt.xlim(200, MAX_HIFF)
    sns.move_legend(fig, 'upper right', bbox_to_anchor=(0.60, 0.8))
    fig.set(title=f'Population HIFF Distribution (baym variants)')
    fig.savefig(OUTPUT_PATH / 'baym_variants'/ f'hiff_dist.png', dpi=600)
    plt.close()
    with open(OUTPUT_PATH / 'baym_variants' / 'mannwitneyu.txt', 'w') as file:
        compare_hiff_distributions(file, head_to_head, 'environment')


if __name__ == '__main__':
    baym_variants()
