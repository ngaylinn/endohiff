"""Run supplemental experiments .
"""
import numpy as np
import polars as pl
import taichi as ti
from compare_experiments import compare_experiments

from constants import ENVIRONMENT_SHAPE, INNER_GENERATIONS, OUTPUT_PATH
from environments import expand_shape, make_field, make_baym
from inner_population import InnerPopulation, get_default_params
from visualize_inner_population import visualize_experiment

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)


def make_stretched(shape=None):
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
    shape = expand_shape(shape)
    return np.broadcast_to(np.expand_dims(ramp_up_and_down, 1), shape)


# This is just a variation on the flat environment, but using the minimum
# fitness threshold that got the best fitness score in a garduated setting.
def make_noramp(shape=None):
    shape = expand_shape(shape)
    return np.full(shape, 320)


def make_gap(shape=None):
    shape = expand_shape(shape)
    steep = np.full(shape, 320)
    steep[0,    :5 ] = 64
    steep[0,   5:10] = 128
    steep[0,  25:30] = 384
    steep[0,  30:34] = 448
    steep[0,  34:39] = 384
    steep[0, -10:-5] = 128
    steep[0,  -5:  ] = 64
    return steep


def make_high(shape=None):
    shape = expand_shape(shape)
    high = np.full(shape, 320)
    high[0,   :25] = 320
    high[0, 25:30] = 384
    high[0, 30:34] = 448
    high[0, 34:39] = 384
    high[0, 39:  ] = 320
    return high


def baym_variants():
    print('Testing effect of tweaking the baym environment.')

    path = OUTPUT_PATH / 'baym_variants'
    environments = {
        'baym': make_baym,
        'gap': make_gap,
        'high': make_high,
    }
    logs = {}

    env = make_field()
    for env_name, make_env in environments.items():
        env.from_numpy(make_env())
        params = get_default_params()
        inner_population = InnerPopulation()
        inner_population.evolve(env, params)
        inner_log = inner_population.get_logs(0)
        logs[env_name] = inner_log

        print()
        print(f'Visualizing results for {env_name}...')
        env_path = OUTPUT_PATH / 'baym_variants' / env_name
        env_path.mkdir(exist_ok=True, parents=True)
        visualize_experiment(env_path, inner_log, env.to_numpy()[0])

    # Merge the logs from the baym and stretched conditions so we can compare
    # them head-to-head.
    head_to_head = pl.concat((
        logs['baym'].with_columns(Environment=pl.lit('baym')),
        logs['gap'].with_columns(Environment=pl.lit('gap')),
        logs['high'].with_columns(Environment=pl.lit('high'))
    )
    #).filter(
    #    pl.col('alive') & (pl.col('Generation') == INNER_GENERATIONS - 1)
    #).group_by(
    #    'environment', 'x', 'y'
    #).agg(
    #    pl.col('fitness').mean().alias('Mean Fitness')
    )
    compare_experiments(path, head_to_head)


if __name__ == '__main__':
    baym_variants()
