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
from environments import (
    ALL_ENVIRONMENT_NAMES, ENV_DTYPE, STATIC_ENVIRONMENTS, Environments)
from inner_population import InnerPopulation, get_default_params
from outer_population import OuterPopulation
from outer_fitness import FitnessEvaluator, get_per_trial_scores

ti.init(ti.cuda, unrolling_limit=0)

SWEEP_SIZE = CARRYING_CAPACITY
TOURNAMENT_SIZES = np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1
MORTALITY_RATES = np.linspace(0, 1, SWEEP_SIZE)
MR_LABELS = [f'{mr:0.3f}' for mr in MORTALITY_RATES]


# TODO: Sweep both environments at once?
def sweep_static(env_name, verbose):
    env = STATIC_ENVIRONMENTS[env_name](SWEEP_SIZE * NUM_TRIALS)
    params = get_default_params(shape=SWEEP_SIZE * NUM_TRIALS)
    inner_population = InnerPopulation(SWEEP_SIZE * NUM_TRIALS)

    if verbose:
        print('Evolving bitstrings for all parameters...')
        tick_progress = trange(SWEEP_SIZE).update
    else:
        tick_progress = lambda: None

    # Sweep over all tournament sizes in parallel.
    params.tournament_size.from_numpy(TOURNAMENT_SIZES.repeat(NUM_TRIALS))

    # Sweep over mortality rate in serial (not enough memory to parallelize)
    frames = []
    for mortality_rate in MORTALITY_RATES:
        params.mortality_rate.fill(mortality_rate)

        # Run NUM_TRIALS iterations in parallel
        inner_population.evolve(env, params)
        scores = get_per_trial_scores(inner_population)
        frames.append(pl.DataFrame({
            'Tournament Size': TOURNAMENT_SIZES.repeat(NUM_TRIALS),
            'Mortality Rate': [mortality_rate] * SWEEP_SIZE * NUM_TRIALS,
            'Fitness': scores,
            'Environment': [env_name] * SWEEP_SIZE * NUM_TRIALS,
        }))
        tick_progress()

    return pl.concat(frames)


def sweep_evolved(verbose):
    outer_population = OuterPopulation(NUM_TRIALS, False)
    SIM_SIZE = NUM_TRIALS * OUTER_POPULATION_SIZE
    inner_population = InnerPopulation(SIM_SIZE)
    params = get_default_params(shape=(SIM_SIZE))
    evaluator = FitnessEvaluator(outer_population=outer_population)

    envs_path = OUTPUT_PATH / 'sweep' / 'cppn' / 'env.npy'

    if not envs_path.exists():
        if verbose:
            print('Evolving environemnts for all parameters...')
            tick_progress = trange(
                SWEEP_SIZE * SWEEP_SIZE * OUTER_GENERATIONS).update
        else:
            tick_progress = lambda: None

        # First, sweep over both params and evolve an environment in each setting,
        # recording the best environment for each configuration.
        # TODO: Make this 2D and index by sweep index to avoid any ambiguity
        # of what env goes with what settings in the second pass.
        best_envs = np.zeros(SWEEP_SIZE * SWEEP_SIZE, ENV_DTYPE)
        i = 0
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
                    tick_progress()
                # TODO: The results actually look really noisy! Occasionally we
                # get abysmal performance for a single configuration, when its
                # neighbors do fine. Perhaps we could be more clever at picking
                # environments from this large population that are more
                # reliable, and not just the ones that did best in one trial.
                env_data = environments.to_numpy()
                best_trial = evaluator.get_best_trial(og)
                best_envs[i] = env_data[best_trial]
                i += 1

        # Save the evolved environments!
        np.save(envs_path, best_envs)
    else:
        best_envs = np.load(envs_path)

    if verbose:
        print('Evolving bitstrings for evolved environments...')
        tick_progress = trange(SWEEP_SIZE).update
    else:
        tick_progress = lambda: None

    # TODO: Break this up into two phases, and refactor to share as much code
    # as possible with the static sweeps.

    # Second, sweep over the params again to re-evolve a fresh population in
    # each environment for NUM_TRIALS independent trials. This ensures there is
    # no bias from the larger population size used whene evolving environments.

    # Grab the environments corresponding to the mortality_rate values that
    # we're computing in parallel, and make NUM_TRIALS copies of each one
    # so we can run trials in parallel, then push to the GPU.
    env = Environments(SIM_SIZE)
    best_envs = best_envs.reshape(SWEEP_SIZE, SWEEP_SIZE)

    # Sweep tournament size in parallel. Note we have to pad out the array to
    # match SIM_SIZE
    params.tournament_size.from_numpy(
        np.resize(
            TOURNAMENT_SIZES.repeat(NUM_TRIALS),
            SIM_SIZE))

    # Sweep mortality rate in serial (not enough memory)
    frames = []
    for m, mortality_rate in enumerate(MORTALITY_RATES):
        params.mortality_rate.fill(mortality_rate)
        # Grab just the environments corresponding to the current mortality
        # rate. That includes one for every tournament size value. Repeat each
        # env NUM_TRIALS times, and pad out the array to match SIM_SIZE.
        curr_envs = best_envs[:, m]
        env.from_numpy(np.resize(curr_envs.repeat(NUM_TRIALS), SIM_SIZE))

        # Run NUM_TRIALS iterations in parallel
        inner_population.evolve(env, params)

        scores = get_per_trial_scores(inner_population)
        scores = scores[:SWEEP_SIZE * NUM_TRIALS]
        frames.append(pl.DataFrame({
            'Tournament Size': TOURNAMENT_SIZES.repeat(NUM_TRIALS),
            'Mortality Rate': [mortality_rate] * SWEEP_SIZE * NUM_TRIALS,
            'Fitness': scores,
            'Environment': ['cppn'] * SWEEP_SIZE * NUM_TRIALS,
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
        if env_name == 'cppn':
            data = sweep_evolved(verbose)
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
