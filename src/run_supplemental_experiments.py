"""Run supplemental experiments .
"""
import numpy as np
import polars as pl
import taichi as ti

from constants import OUTPUT_PATH
from .bitstrings.population import BitstrPopulation, make_params_field
from .environments.util import make_env_field, make_baym
from visualize_bitstr_pop import visualize_experiment


def make_gap(shape=None):
    # TODO: Make this from the baym environment.
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
    # TODO: Make this from the baym environment.
    shape = expand_shape(shape)
    high = np.full(shape, 320)
    high[0,   :25] = 320
    high[0, 25:30] = 384
    high[0, 30:34] = 448
    high[0, 34:39] = 384
    high[0, 39:  ] = 320
    return high


def baym_variants():
    ti.init(ti.cuda)

    print('Testing effect of tweaking the baym environment.')

    environments = {
        'baym': make_baym,
        'gap': make_gap,
        'high': make_high,
    }

    env = make_env_field()
    for env_name, make_env in environments.items():
        env.from_numpy(make_env())
        params = make_params_field()
        bitstr_pop = BitstrPopulation()
        bitstr_pop.evolve(env, params)
        inner_log = bitstr_pop.get_logs(0)

        print()
        print(f'Visualizing results for {env_name}...')
        env_path = OUTPUT_PATH / 'baym_variants' / env_name
        env_path.mkdir(exist_ok=True, parents=True)
        visualize_experiment(env_path, inner_log, env.to_numpy()[0])


if __name__ == '__main__':
    baym_variants()
