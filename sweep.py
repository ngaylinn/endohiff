from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import taichi as ti
from tqdm import trange

from constants import (
    CARRYING_CAPACITY, NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE,
    OUTPUT_PATH)
from environments import ALL_ENVIRONMENT_NAMES, STATIC_ENVIRONMENTS
from inner_population import InnerPopulation, get_default_params
from outer_population import OuterPopulation
from outer_fitness import FitnessEvaluator, get_per_trial_scores

ti.init(ti.cuda, unrolling_limit=0)

SWEEP_SIZE = CARRYING_CAPACITY
TOURNAMENT_SIZES = np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1
MORTALITY_RATES = np.linspace(0, 1, SWEEP_SIZE)
MR_LABELS = [f'{mr:0.3f}' for mr in MORTALITY_RATES]

def sweep_static(env_name, verbose):
    env = STATIC_ENVIRONMENTS[env_name](SWEEP_SIZE)
    params = get_default_params(shape=SWEEP_SIZE)
    inner_population = InnerPopulation(SWEEP_SIZE)

    if verbose:
        tick_progress = trange(SWEEP_SIZE * NUM_TRIALS).update
    else:
        tick_progress = lambda: None

    # Sweep over all tournament sizes in parallel.
    params.tournament_size.from_numpy(TOURNAMENT_SIZES)

    # Sweep over mortality rate in serial (not enough memory to parallelize)
    frames = []
    for mortality_rate in MORTALITY_RATES:
        params.mortality_rate.fill(mortality_rate)

        # Run 5 trials in serial (not enough memory to parallelize)
        for _ in range(NUM_TRIALS):
            inner_population.evolve(env, params)
            scores = get_per_trial_scores(inner_population)
            frames.append(pl.DataFrame({
                'Tournament Size': TOURNAMENT_SIZES,
                'Mortality Rate': [mortality_rate] * SWEEP_SIZE,
                'Fitness': scores,
                'Environment': [env_name] * SWEEP_SIZE,
            }))
            tick_progress()

    return pl.concat(frames)


def sweep_evolved(verbose):
    outer_population = OuterPopulation(NUM_TRIALS, False)
    inner_population = InnerPopulation(NUM_TRIALS * OUTER_POPULATION_SIZE)
    params = get_default_params(shape=(NUM_TRIALS * OUTER_POPULATION_SIZE))
    evaluator = FitnessEvaluator(outer_population=outer_population)

    if verbose:
        tick_progress = trange(
            SWEEP_SIZE * SWEEP_SIZE * OUTER_GENERATIONS).update
    else:
        tick_progress = lambda: None

    sweep_envs = []
    frames = []
    for tournament_size in TOURNAMENT_SIZES:
        params.tournament_size.fill(tournament_size)
        for mortality_rate in MORTALITY_RATES:
            params.mortality_rate.fill(mortality_rate)
            outer_population.randomize()
            for og in range(OUTER_GENERATIONS):
                environments = outer_population.make_environments()
                inner_population.evolve(environments, params)
                evaluator.score_populations(inner_population, og)
                if og + 1 < OUTER_GENERATIONS:
                    outer_population.propagate(og)
            # TODO: Actually, we should re-run each experiment with 5 trials
            # rather than taking the best of 50 like we do here.
            env_data = environments.to_numpy()['min_fitness']
            best_trial = evaluator.get_best_trial(og)
            best_env = evaluator.get_best_per_trial(og)[best_trial]
            best_env_data = env_data[best_env]
            sweep_envs.append(best_env_data)
            score = evaluator.get_fitenss(best_env, og)
            frames.append(pl.DataFrame({
                'Tournament Size': tournament_size,
                'Mortality Rate': mortality_rate,
                'Fitness': score,
                'Environment': 'cppn',
            }))
            tick_progress()

    return pl.concat(frames)


def pivot(data):
    return data.to_pandas().pivot_table(
        index='Tournament Size', columns='Mortality Rate',
        values='Fitness', aggfunc='mean')

def chart(data):
    sns.heatmap(
        data,
        xticklabels=[f'{mr:0.3f}' for mr in MORTALITY_RATES],
        cmap='viridis'
    )


def compare():
    data = {}
    for env_name in ['flat', 'baym', 'cppn']:
        data_filename = OUTPUT_PATH / 'sweep' / env_name / 'sweep.parquet'
        if data_filename.exists():
            data[env_name] = pl.read_parquet(data_filename)

    for env_name1, env_name2 in combinations(data.keys(), 2):
        pt1 = pivot(data[env_name1])
        pt2 = pivot(data[env_name2])
        delta = pt2 - pt1
        sns.heatmap(delta, xticklabels=MR_LABELS, cmap='Spectral')
        plt.suptitle(f'{env_name2} - {env_name1}')
        plt.tight_layout()
        plt.savefig(
            OUTPUT_PATH / 'sweep' / f'{env_name2}_vs_{env_name1}.png', dpi=600)
        plt.close()



def main(env_name, verbose):
    path = OUTPUT_PATH / 'sweep' / env_name
    path.mkdir(exist_ok=True, parents=True)

    data_filename = path / 'sweep.parquet'
    if data_filename.exists():
        data = pl.read_parquet(data_filename)
    else:
        data = sweep_static(env_name, verbose)
        data.write_parquet(data_filename)

    sns.heatmap(pivot(data), xticklabels=MR_LABELS, cmap='viridis')
    plt.suptitle(env_name)
    plt.tight_layout()
    plt.savefig(path / 'sweep.png', dpi=600)
    plt.close()

    compare()

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run a selection param sweep on the chosen environment.')
    parser.add_argument(
        'env_name', type=str, choices=ALL_ENVIRONMENT_NAMES,
        help='Which environment to use for this experiment.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1, choices=[0, 1],
        help='Verbosity level (0 for minimal output)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
