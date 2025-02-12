"""Run supplemental experiments .
"""

from argparse import ArgumentParser
import sys

import numpy as np
import polars as pl
import taichi as ti
from compare_experiments import compare_experiments

from constants import INNER_GENERATIONS, OUTPUT_PATH
from environments import Environments, make_baym, make_flat
from inner_population import InnerPopulation, get_default_params
from visualize_inner_population import visualize_experiment

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

    env = Environments()
    env.min_fitness.from_numpy(
        np.broadcast_to(
            np.expand_dims(ramp_up_and_down, 1),
            env.shape))
    env.weights.fill(1.0)
    return env


# This is just a variation on the flat environment, but using the minimum
# fitness threshold that got the best fitness score in a garduated setting.
def make_noramp():
    env = Environments()
    env.min_fitness.from_numpy(np.full(env.shape, 320))
    env.weights.fill(1.0)
    return env


def baym_variants(verbose):
    if verbose > 0:
        print()
        print('Testing effect of tweaking the baym environment.')

    path = OUTPUT_PATH / 'baym_variants'
    environments = {
        'baym': make_baym,
        'stretched': make_stretched,
        'no_ramp': make_noramp,
    }
    logs = {}

    for env_name, make_env in environments.items():
        env = make_env()
        params = get_default_params()
        inner_population = InnerPopulation()
        inner_population.evolve(env, params)
        inner_log = inner_population.get_logs(0)
        logs[env_name] = inner_log

        if verbose > 0:
            print()
            print(f'Visualizing results for {env_name}...')
        env_path = OUTPUT_PATH / 'baym_variants' / env_name
        env_path.mkdir(exist_ok=True, parents=True)
        visualize_experiment(env_path, inner_log, env.to_numpy()[0], verbose)

    # Merge the logs from the baym and stretched conditions so we can compare
    # them head-to-head.
    head_to_head = pl.concat((
        logs['baym'].with_columns(environment=pl.lit('baym')),
        logs['stretched'].with_columns(environment=pl.lit('stretched'))
    )).filter(
        pl.col('alive') & (pl.col('Generation') == INNER_GENERATIONS - 1)
    ).group_by(
        'environment', 'x', 'y'
    ).agg(
        pl.col('hiff').mean().alias('Mean Hiff')
    )
    compare_experiments(path, head_to_head, 'environment')


# TODO: This doesn't work any more. We need some other way of increasing
# selection pressure, or maybe this demonstration is no longer relevant now
# that the experiment design has changed?
def selection_pressure(verbose):
    if verbose > 0:
        print()
        print('Testing effect of increased selection pressure.')

    path = OUTPUT_PATH / 'selection_pressure'
    environments = {
        'baym': make_baym,
        'flat': make_flat,
    }
    logs = {}
    for env_name, make_env in environments.items():
        env = make_env()
        params = get_default_params()
        inner_population = InnerPopulation()
        inner_population.evolve(env, params)
        inner_log = inner_population.get_logs(0)
        logs[env_name] = inner_log

        if verbose > 0:
            print()
            print(f'Visualizing results for {env_name}...')
        env_path = OUTPUT_PATH / 'selection_pressure' / env_name
        env_path.mkdir(exist_ok=True, parents=True)
        visualize_experiment(env_path, inner_log, env.to_numpy()[0], verbose)

    # Merge the logs from the baym and stretched conditions so we can compare
    # them head-to-head.
    head_to_head = pl.concat((
        logs['baym'].with_columns(environment=pl.lit('baym')),
        logs['flat'].with_columns(environment=pl.lit('flat'))
    )).filter(
        pl.col('alive') & (pl.col('Generation') == INNER_GENERATIONS - 1)
    ).group_by(
        'environment', 'x', 'y'
    ).agg(
        pl.col('hiff').mean().alias('Mean Hiff')
    )
    compare_experiments(path, head_to_head, 'environment')


def main(verbose):
    baym_variants(verbose)
    selection_pressure(verbose)
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run and visualize results for supplemental experiments.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))

