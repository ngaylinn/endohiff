from argparse import ArgumentParser
from itertools import combinations
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import taichi as ti
from tqdm import trange

from constants import (
    CARRYING_CAPACITY, ENVIRONMENT_SHAPE, NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE,
    OUTPUT_PATH)
from environments import ALL_ENVIRONMENT_NAMES, STATIC_ENVIRONMENTS, make_field
from inner_population import InnerPopulation, get_default_params_numpy, Params
from outer_population import OuterPopulation
from outer_fitness import FitnessEvaluator, get_per_trial_scores

ti.init(ti.cuda, unrolling_limit=0)

SWEEP_SIZE = CARRYING_CAPACITY


class Param:
    """Represents a single hyperparameter we might sweep over.
    """
    def __init__(self, key, values, sample_points):
        self.key = key
        self.name = key.title().replace('_', ' ')
        self.values = values
        if values.dtype.kind == 'f':
            self.labels = [f'{val:0.3f}' for val in self.values]
        else:
            self.labels = self.values
        self.sample_points = np.array(sample_points)


# Defines all the hyperparameter sweeps we run for this project.
SWEEPS = {
    'selection': (
        Param('mortality_rate',
              np.linspace(0, 1, SWEEP_SIZE), [0, 15, 22]),
        Param('tournament_size',
              np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1, [0, 2, 12])),
    'ratchet': (
        Param('migration_rate',
              np.linspace(0, 3, SWEEP_SIZE), [1, 3, 12]),
        Param('fertility_rate',
              np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1, [3, 6, 12])),
}


class Sweep:
    """Manages a hyperparameter sweep and breaking work into GPU-sized batches.
    """
    def __init__(self, sims_per_setting, sims_per_batch, sweep_kind, env):
        # How many simulations to run in each hyperparamter configuration.
        self.sims_per_setting = sims_per_setting
        # How many simulations to run at a time on the GPU
        self.sims_per_batch = sims_per_batch
        # The shape and size of the sweep.
        sweep_shape = (SWEEP_SIZE, SWEEP_SIZE, sims_per_setting)
        self.total_sims = int(np.prod(sweep_shape))
        self.num_batches = self.total_sims // self.sims_per_batch

        # To keep things simple, the caller must select a number of simulations
        # per batch that evenly divides the total number of simulations to run.
        assert self.total_sims % self.sims_per_batch == 0

        # If env is the name of a static environment, load it up.
        if isinstance(env, str):
            self.env = STATIC_ENVIRONMENTS[env](sweep_shape)
        # Otherwise, env must be the actual environment data to simulate.
        else:
            self.env = env

        # Prepare to sweep over the parameters indicated by sweep_kind.
        self.param1, self.param2 = SWEEPS[sweep_kind]
        self.params = get_default_params_numpy(sweep_shape)
        self.fill_params()

    def fill_params(self):
        # Enumerate all the parameter settings to sweep over in memory. We do
        # this so that we can decouple the iteration over hyperparameter values
        # from iteration over batches for the GPU. That way, we can have batch
        # sizes that don't neatly align with the parameters we sweep over.
        for i1, v1 in enumerate(self.param1.values):
            for i2, v2 in enumerate(self.param2.values):
                # For the two parameters we're sweeping over, overwrite the
                # default values with the sweep values.
                self.params[i1, i2, :][self.param1.key] = v1
                self.params[i1, i2, :][self.param2.key] = v2

                # Some parameter settings are flagged to collect samples (ie,
                # full simulation data) in those conditions. Label those sample
                # points so that when we break of a batch of work, we know
                # where the sample points in that chunk (if any) are, and how
                # to name the artifacts we collect.
                if all((i1 in self.param1.sample_points,
                        i2 in self.param2.sample_points)):
                    for s in range(self.sims_per_setting):
                        self.params[i1, i2, s]['sample_point'] = (
                            self.param1.labels[i1], self.param2.labels[i2], s)


    def batches(self):
        # A generator that goes through all the parameter settings and
        # associated environments in batches that fit on the GPU.
        for start_index in range(0, self.total_sims, self.sims_per_batch):
            # Pull out a GPU-sized chunk of data.
            end_index = start_index + self.sims_per_batch
            env_data = self.env.reshape(-1, *ENVIRONMENT_SHAPE)[start_index:end_index]
            params_data = self.params.ravel()[start_index:end_index]

            # Collect all the non-null sample points
            sample_points = params_data['sample_point']
            sample_indices = np.nonzero(sample_points)[0]
            samples = zip(sample_indices, sample_points[sample_indices])

            # Yield a summary of what to simulate and record in this batch.
            yield (env_data, params_data, samples)


def bitstr_sweep(sweep_kind, env_name, path):
    # Total sims is SWEEP_SIZE * SWEEP_SIZE * NUM_TRIALS = 5**5, so
    # sims_per_batch must be a power of 5 to divide it evenly.
    sims_per_batch = 25

    # Set up an environment for running these simulations.
    env = make_field(sims_per_batch)
    params = Params.field(shape=sims_per_batch)
    inner_population = InnerPopulation(sims_per_batch)

    # Sweep through hyperparameter settings one batch at a time.
    print('Evolving bitstrings for all parameters...')
    sweep = Sweep(NUM_TRIALS, sims_per_batch, sweep_kind, env_name)
    progress = trange(sweep.num_batches)
    frames = []
    for (env_data, params_data, samples) in sweep.batches():
        # Copy this batch of data to the GPU and simulate!
        env.from_numpy(env_data)
        params.from_numpy(params_data)
        inner_population.evolve(env, params)

        # Copy fitness scores to the GPU and add them to the logs along with
        # the hyperparamter settings that correspond to that result.
        frames.append(pl.DataFrame({
            sweep.param1.name: params_data[sweep.param1.key],
            sweep.param2.name: params_data[sweep.param2.key],
            'Fitness': get_per_trial_scores(inner_population),
            'Environment': [env_name] * sims_per_batch,
        }))

        # For all the sample points in this batch...
        for e, (p1_label, p2_label, trial) in samples:
            # Create a sub-directory for this hyperparameter setting.
            param_summary = (
                f'{sweep.param1.key}_{p1_label}_{sweep.param2.key}_{p2_label}')
            sample_path = path / param_summary / env_name / f'trial_{trial}'
            sample_path.mkdir(exist_ok=True, parents=True)

            # Record the environment and the log of bitstring evolution so we
            # know exactly what happened here.
            np.save(sample_path / 'env.npy', env_data[e])
            inner_log = inner_population.get_logs(e)
            inner_log.write_parquet(sample_path / f'inner_log.parquet')

        # Update the progress bar...
        progress.update()

    return pl.concat(frames)


# TODO: Rewrite this entirely in the same style as above.
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


def pivot(sweep_kind, data):
    p1_name, p2_name = (param.name for param in SWEEPS[sweep_kind])
    # Sadly, Polars' pivot method isn't compatible with Seaborn, so use Pandas.
    return data.to_pandas().pivot_table(
        index=p2_name, columns=p1_name, values='Fitness', aggfunc='mean')


def labels(sweep_kind):
    return {
        'xticklabels': SWEEPS[sweep_kind][0].labels,
        'yticklabels': SWEEPS[sweep_kind][1].labels,
    }


def draw_sample_points(sweep_kind):
    x_points = SWEEPS[sweep_kind][0].sample_points + 0.5
    y_points = SWEEPS[sweep_kind][1].sample_points + 0.5
    x_points, y_points = (
        x_points.repeat(len(y_points)),
        np.tile(y_points, len(x_points)))
    plt.plot(x_points, y_points, 'kx', ms=10)


def compare(sweep_kind, path):
    data = {}
    for env_name in ['flat', 'baym', 'cppn']:
        data_filename = path / f'{env_name}.parquet'
        if data_filename.exists():
            data[env_name] = pl.read_parquet(data_filename)

    for env_name1, env_name2 in combinations(data.keys(), 2):
        pt1 = pivot(sweep_kind, data[env_name1])
        pt2 = pivot(sweep_kind, data[env_name2])
        delta = pt2 - pt1
        sns.heatmap(delta, **labels(sweep_kind), cmap='Spectral')
        draw_sample_points(sweep_kind)
        plt.suptitle(f'{env_name2} - {env_name1}')
        plt.tight_layout()
        plt.savefig(path / f'{env_name2}_vs_{env_name1}.png', dpi=600)
        plt.close()


def main(env_name, sweep_kind):
    path = OUTPUT_PATH / f'{sweep_kind}_sweep'
    path.mkdir(exist_ok=True, parents=True)

    data_filename = path / f'{env_name}.parquet'
    if data_filename.exists():
        data = pl.read_parquet(data_filename)
    else:
        if env_name == 'cppn':
            data = env_sweep()
        else:
            data = bitstr_sweep(sweep_kind, env_name, path)
        data.write_parquet(data_filename)

    sns.heatmap(pivot(sweep_kind, data), **labels(sweep_kind), cmap='viridis')
    draw_sample_points(sweep_kind)
    plt.suptitle(env_name)
    plt.tight_layout()
    plt.savefig(path / f'{env_name}.png', dpi=600)
    plt.close()

    compare(sweep_kind, path)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run a selection param sweep on the chosen environment.')
    parser.add_argument(
        'env_name', type=str, choices=ALL_ENVIRONMENT_NAMES,
        help='Which environment to use for this experiment.')
    parser.add_argument(
        'sweep_kind', type=str, choices=SWEEPS.keys(),
        help='Which set of hyperparameters to sweep over.')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
