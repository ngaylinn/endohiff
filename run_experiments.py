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

# TODO: Once we add an evolved environment, include it here, also.
CONDITION_NAMES = ENVIRONMENTS.keys()

def print_summary(name, expt_data, migration, crossover):
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


def get_variant_name(migration, crossover):
    return f'migration_{migration}_crossover_{crossover}'


def summarize_hiff_scores(inner_log, env_name, variant_name):
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


def run_experiments():
    OUTPUT_PATH.mkdir(exist_ok=True)

    total_runs = 2 * 2 * len(ENVIRONMENTS) * NUM_REPETITIONS
    progress = trange(total_runs)
    for migration in [True, False]:
        for crossover in [True, False]:
            variant_name = get_variant_name(migration, crossover)
            subfolder_path = OUTPUT_PATH / variant_name
            subfolder_path.mkdir(exist_ok=True)

            for env_name, make_environment in ENVIRONMENTS.items():
                path = subfolder_path / env_name
                path.mkdir(exist_ok=True)

                inner_population = InnerPopulation()
                environment = make_environment()
                np.savez(path / 'env.npz', **environment.to_numpy())

                hiff_score_frames = []
                # whole_pop_metrics_frames = []
                best_trial_data = None
                best_trial_fitness = 0
                for _ in range(NUM_REPETITIONS):
                    inner_log, whole_pop_metrics, outer_fitness = evolve(
                        inner_population, environment, migration, crossover)
                    if outer_fitness > best_trial_fitness:
                        best_trial_data = inner_log
                    # Keep the per-location mean hiff scores from the last
                    # generation. We'll concatenate these for all trials to get
                    # a sense of the overall performance of the EA in the given
                    # configuration.
                    hiff_score_frames.append(
                        summarize_hiff_scores(inner_log, env_name, variant_name))
                    # whole_pop_metrics_frames.append(
                    #     whole_pop_metrics.with_columns(trial=pl.lit(trial)))
                    progress.update()

                best_trial_data.write_parquet(path / 'best_trial.parquet')
                pl.concat(hiff_score_frames).write_parquet(path / 'hiff_scores.parquet')


if __name__ == '__main__':
    run_experiments()
