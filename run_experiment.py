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
import polars as pl
import taichi as ti
from tqdm import trange

from constants import (
    INNER_GENERATIONS, NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE,
    OUTPUT_PATH)
from environments import ALL_ENVIRONMENT_NAMES, STATIC_ENVIRONMENTS
from inner_population import InnerPopulation
from outer_population import OuterPopulation

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)


def get_variant_name(migration, crossover):
    return f'migration_{migration}_crossover_{crossover}'


def get_all_trials():
    """A generator summarizing all experiment trials run by this script.
    """
    for migration in [True, False]:
        for crossover in [True, False]:
            variant_name = get_variant_name(migration, crossover)
            for env_name in ALL_ENVIRONMENT_NAMES:
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
    os.symlink(
        os.path.abspath(path / f'trial{best_index}'), best_trial_link,
        target_is_directory=True)


def score_populations(all_logs):
    """Score how well the inner populations evolved in this configuration.

    This is used to find the "best" run across multiple trials, and to evolve
    environments for inner populations to evolve within.
    """
    scores = []
    for inner_log in all_logs:
        score = inner_log.filter(
            # Look only at live individuals in the last generation.
            (pl.col('id') > 0) &
            (pl.col('Generation') == INNER_GENERATIONS - 1)
        # Find the average hiff score in every location of the environment.
        ).group_by(
            'x', 'y'
        ).agg(
            pl.col('hiff').mean()
        # Now take the median location-level hiff score per environment, as a
        # proxy for how heavily the distribution is weighted towards high scores.
        )['hiff'].median()

        # Handle missing values gracefully.
        scores.append(0.0 if score is None else float(score))

    return np.array(scores)


def run_experiment_static_env(env_name, migration, crossover):
    variant_name = get_variant_name(migration, crossover)
    path = OUTPUT_PATH / variant_name / env_name

    # Evolve a population of bitstrings in a static environment NUM_TRIALS
    # times, in parallel.
    environments = STATIC_ENVIRONMENTS[env_name](NUM_TRIALS)
    inner_population = InnerPopulation(NUM_TRIALS)
    inner_population.evolve(environments, migration, crossover)

    # Save a full summary for each trial.
    all_logs = inner_population.get_logs()
    for t, inner_log in enumerate(all_logs):
        trial_path = path / f'trial{t}'
        trial_path.mkdir(exist_ok=True, parents=True)
        # Save a copy of the environment with every trial even though it
        # doesn't change, just for consistency with the evolved environment
        # outputs.
        np.savez(trial_path / 'env.npz', **environments[t])
        inner_log.write_parquet(trial_path / 'inner_log.parquet')

    # Create a symlink to indicate which trial was the best one.
    best_index = score_populations(all_logs).argmax()
    link_best_trial(path, best_index)


def run_experiment_evolved_env(migration, crossover, verbose):
    variant_name = get_variant_name(migration, crossover)
    path = OUTPUT_PATH / variant_name / 'cppn'

    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        print('Evolving environments...')
        tick_progress = trange(NUM_TRIALS * OUTER_GENERATIONS).update
    else:
        tick_progress = lambda: None

    # Evovle a population of CPPN environments NUM_TRIALS times.
    outer_population = OuterPopulation()
    inner_population = InnerPopulation(OUTER_POPULATION_SIZE)
    best_trial_fitness = -1
    best_trial_index = -1
    for t in range(NUM_TRIALS):
        # Evolve an outer population of environments.
        outer_population.randomize()
        for og in range(OUTER_GENERATIONS):
            environments = outer_population.make_environments()
            inner_population.evolve(environments, migration, crossover)
            outer_fitness = score_populations(inner_population.get_logs())
            if og + 1 < OUTER_GENERATIONS:
                outer_population.propagate(outer_fitness, og)
            tick_progress()

        # Track the best evolved environment for this trial and all trials.
        best_env = outer_fitness.argmax()
        best_fitness = outer_fitness[best_env]
        if best_fitness > best_trial_fitness:
            best_trial_fitness = best_fitness
            best_trial_index = t

        # Setup the output directory.
        trial_path = path / f'trial{t}'
        trial_path.mkdir(exist_ok=True, parents=True)

        # Find which evolved environment from the last trial evolved the best
        # and save a visualization of that environment.
        np.savez(trial_path / 'env.npz', **environments[best_env])

        # Save summaries of the final inner population for the best environment
        # from each trial.
        inner_log = inner_population.get_logs(best_env)
        inner_log.write_parquet(trial_path / 'inner_log.parquet')

        # Save the fitness history for the full outer population in each trial.
        outer_log = outer_population.get_logs()
        outer_log.write_parquet(trial_path / 'outer_log.parquet')

    # Create a symlink to indicate which trial was the best one.
    link_best_trial(path, best_trial_index)


def main(env_name, migration, crossover, verbose):
    if env_name == 'cppn':
        run_experiment_evolved_env(migration, crossover, verbose)
    else:
        run_experiment_static_env(env_name, migration, crossover)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run a single experiment and record results.')
    parser.register('type', 'bool string', lambda s: s == 'True')
    parser.add_argument(
        'env_name', type=str, help='Which environment to use for this experiment.')
    parser.add_argument(
        'migration', type='bool string',
        help='Whether to enable migration for this experiment (True or False).')
    parser.add_argument(
        'crossover', type='bool string',
        help='Whether to enable crossover for this experiment (True or False).')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    # Make sure the specified environment is recognized.
    if not args['env_name'] in ALL_ENVIRONMENT_NAMES:
        raise ValueError(f'env_name must be one of: {ALL_ENVIRONMENT_NAMES}')

    sys.exit(main(**args))
