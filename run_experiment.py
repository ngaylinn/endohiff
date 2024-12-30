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

from constants import (
    CARRYING_CAPACITY, DEAD_ID, ENVIRONMENT_SHAPE, INNER_GENERATIONS,
    NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE, OUTPUT_PATH)
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


# A flat list of mean HIFF scores sized to fit one value for each location in
# the environment, used by compute_mean_hiffs and score_populations.
local_hiff_means = ti.field(float, shape=np.prod(ENVIRONMENT_SHAPE))

@ti.kernel
def compute_mean_hiffs(inner_population: ti.template(), e: int):
    g = INNER_GENERATIONS - 1
    for x, y in ti.ndrange(*ENVIRONMENT_SHAPE):
        hiff_sum = ti.cast(0, ti.uint32)
        alive_count = 0
        for i in range(CARRYING_CAPACITY):
            individual = inner_population.pop[e, g, x, y, i]
            if individual.id != DEAD_ID:
                hiff_sum += individual.hiff
                alive_count += 1
        m = x * ENVIRONMENT_SHAPE[1] + y
        local_hiff_means[m] = ti.select(
            alive_count > 0, hiff_sum / alive_count, 0.0)


def score_populations(inner_population, fitness=None, generation=None):
    """Score how well the inner populations evolved in this configuration.

    This is used to find the "best" run across multiple trials, and to evolve
    environments for inner populations to evolve within.
    """
    best_index = -1
    best_score = -1
    # Unfortunately, the only off-the-shelf sorting algorithm for Taichi only
    # supports one-dimensional arrays, so score each environment one at a time.
    for e in range(OUTER_POPULATION_SIZE):
        # Find the average hiff score for each environment location, then sort
        # that list and grab the middle value to find the median. We use that
        # as a proxy for the overall distribution skewing towards a high score.
        compute_mean_hiffs(inner_population, e)
        ti.algorithms.parallel_sort(local_hiff_means)
        score = local_hiff_means[local_hiff_means.shape[0] // 2]

        # If we're tracking fitness scores in a field, store the result there.
        if fitness is not None and generation is not None:
            fitness[generation, 0, e] = score

        # Keep track of the highest scoring population of all the ones scored.
        if score > best_score:
            best_index = e
            best_score = score

    # Return the index and score of the best inner population.
    return best_index, best_score


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
        np.save(trial_path / 'env.npy', environments.to_numpy()[t])
        inner_log.write_parquet(trial_path / 'inner_log.parquet')
        # Create an empty outer log, just for consistency with the evolved
        # environment experiments.
        (trial_path / 'outer_log.parquet').touch()

    # Create a symlink to indicate which trial was the best one.
    best_index, _ = score_populations(inner_population)
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

    # Evolve a population of CPPN environments NUM_TRIALS times.
    outer_population = OuterPopulation()
    inner_population = InnerPopulation(OUTER_POPULATION_SIZE)
    best_trial_fitness = -1
    best_trial_index = -1
    outer_fitness = ti.field(float, OUTER_POPULATION_SIZE)
    for t in range(NUM_TRIALS):
        # Evolve an outer population of environments. For performance reason,
        # this entire loop runs on the GPU with minimal data transfers.
        outer_population.randomize()
        for og in range(OUTER_GENERATIONS):
            environments = outer_population.make_environments()
            inner_population.evolve(environments, migration, crossover)
            best_env_index, best_env_fitness = score_populations(
                inner_population, outer_population.matchmaker.fitness, og)
            if og + 1 < OUTER_GENERATIONS:
               outer_population.propagate(og)
            tick_progress()

        # Track the best evolved environment for all trials.
        if best_env_fitness > best_trial_fitness:
            best_trial_fitness = best_env_fitness
            best_trial_index = t

        # Setup the output directory.
        trial_path = path / f'trial{t}'
        trial_path.mkdir(exist_ok=True, parents=True)

        # Save the evolved environment from this trial that produced the best
        # evolving inner population.
        np.save(trial_path / 'env.npy', environments.to_numpy()[best_env_index])

        # Save summaries of the final inner population for the best environment
        # from each trial.
        inner_log = inner_population.get_logs(best_env_index)
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
