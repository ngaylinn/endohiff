"""Run the main series of evolutionary experiments for this project.

Calling this module from the command line will evolve a population of
bitstrings across a series of designed environments logging the full results.
It will also evolve an environment, using the fitness of a population evolved
within that environment as a sign of fitness for that environment. All of these
experiments are run with and without crossover and migration enabled to measure
the impact of these variations on the evolutionary dynamics.
"""

from argparse import ArgumentError, ArgumentParser
import sys

import numpy as np
import polars as pl
import taichi as ti
from tqdm import trange

from constants import INNER_GENERATIONS, MAX_HIFF, NUM_REPETITIONS, OUTPUT_PATH
from environment import ENVIRONMENTS
from evolve import evolve
from inner_population import InnerPopulation

# We store weights in a vector, which Taichi warns could cause slow compile
# times. In practice, this doesn't seem like a problem, so disable the warning.
ti.init(ti.cuda, unrolling_limit=0)


def print_summary(name, expt_data, migration, crossover):
    """For debugging, print a brief summary of a single experiment."""
    best_hiff, best_bitstr = expt_data.filter(
        pl.col('hiff') == pl.col('hiff').max()
    ).select(
        'hiff', 'bitstr'
    )
    print(f'Experiment condition: {name}')
    print(f'Migration: {migration}, Crossover: {crossover}')
    print(f'{len(best_hiff)} individual(s) found the highest score '
          f'({best_hiff[0]} out of a possible {MAX_HIFF})')
    print(f'Example: {best_bitstr[0]:064b}')
    print()


def summarize_hiff_scores(inner_log, env_name, variant_name):
    """Reduce logs data to just a summary of fitness over time.

    Since fitness is fundamentally spatial in this experiment, record the
    distribution of local mean fitness scores in each generation. Make sure
    it's annotated with the environment and variant name in order to compare
    across experimental conditions.
    """
    # Restrict to living individuals in the final generation.
    return inner_log.filter(
        (pl.col('id') > 0) &
        (pl.col('Generation') == INNER_GENERATIONS - 1)
    # Average hiff scores for each location in the
    # environment, so we can analyze relative concentrations
    ).group_by(
        'x', 'y'
    ).agg(
        pl.col('hiff').mean().alias('Mean Hiff')
    # Add metadata for slicing results.
    ).with_columns(
        environment=pl.lit(env_name),
        variant=pl.lit(variant_name)
    # Drop the location data. We just want to make a histogram over locations
    # in the environment, so we need samples from each location, but don't need
    # to remember where the samples came from.
    ).drop(
        'x', 'y'
    )


def main(env, migration, crossover, verbose):
    # Setup output directory and files
    variant_name = f'migration_{migration}_crossover_{crossover}'
    path = OUTPUT_PATH / variant_name / env
    path.mkdir(exist_ok=True, parents=True)
    env_file = path / 'env.npz'
    best_trial_file = path / 'best_trial.parquet'
    hiff_scores_file = path / 'hiff_scores.parquet'

    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        repetitions = trange(NUM_REPETITIONS)
    else:
        repetitions = range(NUM_REPETITIONS)

    # Generate the population and environment. Save a copy of the environment
    # for posterity (this will beimportant for evolved environments)
    inner_population = InnerPopulation()
    environment = ENVIRONMENTS[env]()
    np.savez(env_file, **environment.to_numpy())

    # Aggregate and track hiff score data across all trials to compare variants
    # and environments with each other, but also remember the full details of
    # the best trial to visualize what happened.
    hiff_score_frames = []
    best_trial_data = None
    best_trial_fitness = 0
    for _ in repetitions:
        # Actually run the experiment.
        inner_log, outer_fitness = evolve(
            inner_population, environment, migration, crossover)

        # Track the results.
        if outer_fitness > best_trial_fitness:
            best_trial_data = inner_log
        hiff_score_frames.append(
            summarize_hiff_scores(inner_log, env, variant_name))

    # Once all trials for this experiment have run, save the results to disk.
    best_trial_data.write_parquet(best_trial_file)
    pl.concat(hiff_score_frames).write_parquet(hiff_scores_file)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run a single experiment and record results.')
    parser.register('type', 'bool string', lambda s: s == 'True')
    parser.add_argument(
        'env', type=str, help='Which environment to use for this experiment.')
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
    if not args['env'] in ENVIRONMENTS:
        raise ValueError('Unrecognized environment name.')

    # Actually render these results.
    sys.exit(main(**args))
