"""Run the main series of evolutionary experiments for this project.

Calling this module from the command line will evolve a population of
bitstrings across a series of designed environments logging the full results.
It will also evolve an environment, using the fitness of a population evolved
within that environment as a sign of fitness for that environment. All of these
experiments are run with and without crossover and migration enabled to measure
the impact of these variations on the evolutionary dynamics.
"""

from argparse import ArgumentParser
import os
import sys

import numpy as np
import taichi as ti
from tqdm import trange

from .constants import (
    ENV_NAMES, NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE, OUTPUT_PATH)
from .bitstrings.population import BitstrPopulation, make_params_field
from .environments.util import STATIC_ENVIRONMENTS, make_env_field
from .environments.population import EnvironmentPopulation
from .environments.fitness import EnvFitnessEvaluator, get_best_trial


def get_variant_name(migration, crossover):
    return f'migration_{migration}_crossover_{crossover}'


def get_all_trials():
    """A generator summarizing all experiment trials run by this script.
    """
    for migration in [True, False]:
        for crossover in [True, False]:
            variant_name = get_variant_name(migration, crossover)
            for env_name in ENV_NAMES:
                for trial in range(NUM_TRIALS):
                    log_file = (
                        OUTPUT_PATH / variant_name / env_name /
                        f'trial{trial}' / 'inner_log.parquet')
                    yield (variant_name, env_name, trial, log_file)


def link_best_trial(path, best_index):
    best_trial_link = path / 'best_trial'

    # Remove the old best trial symlink, if it exists.
    try:
        os.unlink(best_trial_link)
    except FileNotFoundError:
        pass

    # Create a symlink to indicate which trial was the best one.
    # TODO: This doesn't work when copying data across file systems. Find a
    # better way to track the best trial?
    os.symlink(
        os.path.abspath(path / f'trial{best_index}'), best_trial_link,
        target_is_directory=True)


def run_experiment_static_env(env_name, migration, crossover):
    variant_name = get_variant_name(migration, crossover)
    path = OUTPUT_PATH / variant_name / env_name

    # Set up environments and parameters for the BitstrPopulation.
    environments = make_env_field(NUM_TRIALS)
    environments.from_numpy(STATIC_ENVIRONMENTS[env_name](NUM_TRIALS))
    params = make_params_field(NUM_TRIALS)
    if not migration:
        params.migration_rate.fill(0.0)
    if not crossover:
        params.crossover_rate.fill(0.0)

    # Evolve a population of bitstrings in a static environment NUM_TRIALS
    # times, in parallel.
    bitstr_pop = BitstrPopulation(NUM_TRIALS)
    bitstr_pop.evolve(environments, params)

    # Save a full summary for each trial.
    env_data = environments.to_numpy()
    for t in range(NUM_TRIALS):
        trial_path = path / f'trial{t}'
        trial_path.mkdir(exist_ok=True, parents=True)
        # Save a copy of the environment with every trial even though it
        # doesn't change, just for consistency with the evolved environment
        # outputs.
        np.save(trial_path / 'env.npy', env_data[t])
        inner_log = bitstr_pop.get_logs(t)
        inner_log.write_parquet(trial_path / 'inner_log.parquet')

    # Create an empty outer log, just for consistency with the evolved
    # environment experiments.
    (path / 'outer_log.parquet').touch()

    # Create a symlink to indicate which trial was the best one.
    link_best_trial(path, get_best_trial(bitstr_pop))


def run_experiment_evolved_env(migration, crossover, verbose):
    variant_name = get_variant_name(migration, crossover)
    path = OUTPUT_PATH / variant_name / f'cppn'

    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        print('Evolving environments...')
        progress = trange(OUTER_GENERATIONS)
    else:
        progress = range(OUTER_GENERATIONS)

    # Set up parameters for the BitstrPopulation.
    params = make_params_field(NUM_TRIALS)
    #if not migration:
    #    params.migration_rate.fill(0.0)
    #if not crossover:
    #    params.crossover_rate.fill(0.0)

    # Evolve a population of CPPN environments NUM_TRIALS times, and a whole
    # population of bitstrings within each one.
    env_pop = EnvironmentPopulation(NUM_TRIALS)
    bitstr_pop = BitstrPopulation(NUM_TRIALS * OUTER_POPULATION_SIZE)
    evaluator = EnvFitnessEvaluator(env_pop=env_pop)

    # Evolve an outer population of environments. For performance reason,
    # this entire loop runs on the GPU with minimal data transfers.
    env_pop.randomize()
    for og in progress:
        environments = env_pop.make_environments()
        bitstr_pop.evolve(environments, params)
        evaluator.score_pops(bitstr_pop, og)
        if og + 1 < OUTER_GENERATIONS:
            env_pop.propagate(og)

    # Log the best bitstr_pop from each trial from the last outer
    # generation (ie, a fully evolved CPPN environment)
    env_data = environments.to_numpy()
    best_env_per_trial = evaluator.get_best_per_trial(og)
    for t in range(NUM_TRIALS):
        trial_path = path / f'trial{t}'
        trial_path.mkdir(exist_ok=True, parents=True)

        # Save the best environment and logs associated with this trial.
        e = best_env_per_trial[t]
        np.save(trial_path / 'env.npy', env_data[e])
        inner_log = bitstr_pop.get_logs(e)
        inner_log.write_parquet(trial_path / 'inner_log.parquet')

    # Save the full logs for evolving the outer population.
    outer_log = env_pop.get_logs()
    outer_log.write_parquet(path / 'outer_log.parquet')

    # Create a symlink to indicate which trial was the best one.
    link_best_trial(path, evaluator.get_best_trial(og))


def main(env_name, migration, crossover, verbose):
    ti.init(ti.cuda)

    if env_name == 'cppn':
        run_experiment_evolved_env(migration, crossover, verbose)
    else:
        run_experiment_static_env(env_name, migration, crossover)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run a single experiment and record results.')
    # NOTE: We use strings instead of bools because that's convenient when
    # interfacing with Snakemake.
    parser.register('type', 'bool string', lambda s: s == 'True')
    parser.add_argument(
        'env_name', type=str, choices=ALL_ENVIRONMENT_NAMES,
        help='Which environment to use for this experiment.')
    parser.add_argument(
        'migration', type='bool string', choices=[True, False],
        help='Whether to enable migration for this experiment.')
    parser.add_argument(
        'crossover', type='bool string', choices=[True, False],
        help='Whether to enable crossover for this experiment.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1, choices=[0, 1],
        help='Verbosity level (0 for minimal output)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
