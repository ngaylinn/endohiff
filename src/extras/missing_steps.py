"""Evolve bitstrings in variations of the baym environment, with missing steps.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import taichi as ti

from src.bitstrings.population import BitstrPopulation, make_params_field
from src.bitstrings.visualize_population import save_fitness_map
from src.environments.util import make_env_field, make_baym
from src.environments.visualize_one import save_env_map


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


MISSING_STEP_ENVS = {
    'baym': make_baym,
    'gap':  make_gap,
    'high': make_high,
}


def main(path):
    ti.init(ti.cuda)
    path.mkdir(exist_ok=True, parents=True)

    # Setup memory allocations on the GPU.
    env_field = make_env_field()
    params_field = make_params_field()
    bitstr_pop = BitstrPopulation()
    for env_name, make_env in MISSING_STEP_ENVS.items():
        # Render the environment and save a visualization.
        env_data = make_env()
        save_env_map(env_data, path / f'{env_name}_env_map.png')

        # Push the environment to device, and evolve some bitstrings.
        env_field.from_numpy(env_data)
        bitstr_pop.evolve(env_field, params_field)

        # Save a visualization of the final simulation state.
        bitstr_log = bitstr_pop.get_logs(0)
        save_fitness_map(bitstr_log, path / f'{env_name}_fitness_map.png')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evolve bitstrings in variants of the baym environment')
    parser.add_argument(
        'path', type=Path,
        help='Where to save visualizations from this side experiment')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
