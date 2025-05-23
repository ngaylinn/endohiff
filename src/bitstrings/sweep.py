"""Evolve bitstrings for all hyperparmeter values, in a given environment.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import polars as pl
import taichi as ti
from tqdm import trange

from src.bitstrings.population import make_params_field, BitstrPopulation
from src.constants import ENV_NAMES, NUM_TRIALS
from src.environments.fitness import get_per_trial_env_fitness
from src.environments.util import STATIC_ENVIRONMENTS, make_env_field
from src.sweep import SWEEP_KINDS, SWEEP_SIZE, SWEEP_SHAPE, Sweep


def main(path, sweep_kind, env_name):
    ti.init(ti.cuda)

    # If requested, load evolved environments from disk.
    if env_name == 'cppn':
        env_data = np.load(path / 'cppn_envs.npy')
    # Otherwise, use one of the pre-made static definitions.
    else:
        env_data = STATIC_ENVIRONMENTS[env_name](SWEEP_SHAPE)

    # Set up an environment for running these simulations. The workstation we
    # used has enough GPU memory to run all the trials for one whole dimension
    # of the hyperparameter sweep in parallel, so this is what we do.
    sweep = Sweep(sweep_kind)
    batch_size = SWEEP_SIZE * NUM_TRIALS
    env_field = make_env_field(batch_size)
    params_field = make_params_field(batch_size)
    bitstr_pop = BitstrPopulation(batch_size)

    # Sweep through hyperparameter settings one batch at a time.
    print('Evolving bitstrings for all parameters...')
    progress = trange(SWEEP_SIZE)
    frames = []
    for params_data, i1, samples in sweep.iter_batched():
        # Load the appropriate set of environments and parameter settings to
        # the GPU, then evolve a bitstring population.
        env_field.from_numpy(env_data[i1].repeat(NUM_TRIALS, axis=0))
        params_field.from_numpy(params_data.repeat(NUM_TRIALS))
        bitstr_pop.evolve(env_field, params_field)

        # Pull fitness scores from the GPU and add them to the logs along with
        # the hyperparamter settings that correspond to those results.
        frames.append(pl.DataFrame({
            sweep.param1.name: params_data[sweep.param1.key].repeat(NUM_TRIALS),
            sweep.param2.name: params_data[sweep.param2.key].repeat(NUM_TRIALS),
            'Fitness': get_per_trial_env_fitness(bitstr_pop),
        }))

        # For all the sample points in this batch...
        for i2 in samples:
            for t in range(NUM_TRIALS):
                # This whole batch corresponds to one value of i1, but several
                # values of i2, each repeated NUM_TRIALS times. So, compute the
                # environment index from the parameter and trial indices.
                e = i2 * NUM_TRIALS + t

                # Record the environment and the full bitstring evolution log
                # for all trials with these hyperparameter settings.
                sample_path = (
                    path / sweep.summary(i1, i2) / env_name / f'trial_{t}')
                sample_path.mkdir(exist_ok=True, parents=True)
                np.save(sample_path / 'env.npy', env_data[i1, i2])
                bitstr_log = bitstr_pop.get_logs(e)
                bitstr_log.write_parquet(sample_path / f'bitstr_log.parquet')

        progress.update()

    sweep_log = pl.concat(frames)
    sweep_log.write_parquet(path / f'{env_name}.parquet')

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evolve bitstrings across many hyperparameter settings.')
    parser.add_argument(
        'path', type=Path,
        help='Path for all sweep data')
    parser.add_argument(
        'sweep_kind', type=str, choices=SWEEP_KINDS,
        help=f'Which kind of sweep to run (one of {SWEEP_KINDS})')
    parser.add_argument(
        'env_name', type=str, choices=ENV_NAMES,
        help=f'Which environment to use (one of {ENV_NAMES})')
    args = vars(parser.parse_args())
    sys.exit(main(**args))

