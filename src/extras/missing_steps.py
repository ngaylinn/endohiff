from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import taichi as ti

from ..bitstrings.population import BitstrPopulation, make_params_field
from ..bitstrings.visualize_population import save_fitness_map
from ..environments.util import make_env_field, make_baym
from ..environments.visualize_one import save_env_map


def make_gap():
    steep = np.ascontiguousarray(make_baym())
    steep[:,  10: 25, :] = 320
    steep[:, -25:-10, :] = 320
    return steep


def make_high():
    high = np.ascontiguousarray(make_baym())
    high[:, :25,  :] = 320
    high[:, -25:, :] = 320
    return high


def main(path):
    ti.init(ti.cuda)

    environments = {
        'baym': make_baym,
        'gap': make_gap,
        'high': make_high,
    }

    env = make_env_field()
    for env_name, make_env in environments.items():
        env_data = make_env()
        save_env_map(env_data, path / f'{env_name}_env_map.png')

        env.from_numpy(env_data)
        params = make_params_field()
        bitstr_pop = BitstrPopulation()
        bitstr_pop.evolve(env, params)
        inner_log = bitstr_pop.get_logs(0)
        save_fitness_map(inner_log, path / f'{env_name}_fitness_map.png')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evolve bitstrings in variants of the baym environment')
    parser.add_argument(
        'path', type=Path,
        help='Where to save visualizations from this side experiment')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
