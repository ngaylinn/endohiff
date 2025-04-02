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
    CARRYING_CAPACITY, NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE,
    OUTPUT_PATH)
from environments import (
    ALL_ENVIRONMENT_NAMES, STATIC_ENVIRONMENTS, make_field, make_flat)
from inner_population import InnerPopulation, get_default_params_numpy, Params
from outer_population import OuterPopulation
from outer_fitness import FitnessEvaluator, get_per_trial_scores

SWEEP_SIZE = CARRYING_CAPACITY
SWEEP_SHAPE = (SWEEP_SIZE, SWEEP_SIZE)
# SWEEP_KINDS = ['selection', 'ratchet']
SWEEP_KINDS = ['selection']

def all_sweep_sample_dirs():
    summaries = []
    for sweep_kind in SWEEP_KINDS:
        summaries.extend(
            [f'{sweep_kind}_sweep/{summaries}' for summaries in
             Sweep(sweep_kind).enumerate_sample_summaries()])
    return summaries


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


class Sweep:
    """Represents a 2D sweep over a pair of hyperparameters.
    """
    def __init__(self, sweep_kind):
        # The sweep configurations used by this project:
        if sweep_kind == 'selection':
            self.param1 = Param(
                'mortality_rate',
                np.linspace(0, 1, SWEEP_SIZE), [0, 15, 22])
            self.param2 = Param(
                'tournament_size',
                np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1, [0, 2, 12])
        elif sweep_kind == 'ratchet':
            self.param1 = Param(
                'migration_rate',
                np.linspace(0, 3, SWEEP_SIZE), [1, 3, 12])
            self.param2 = Param(
                'fertility_rate',
                np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1, [3, 6, 12])

    def labels(self):
        # Format param values for rendering in a Seaborn heatmap
        return {
            'xticklabels': self.param1.labels,
            'yticklabels': self.param2.labels,
        }

    def sample_points(self):
        # Each hyperparameter has a number of sample points associated with it.
        # We record the full history of simulations with those hyperparameters.
        # This function summarizes all the sample points in the 2D space
        # defined by both hyperparameters in the sweep.
        x_points = self.param1.sample_points
        y_points = self.param2.sample_points
        return np.array((
            x_points.repeat(len(y_points)),
            np.tile(y_points, len(x_points))))

    def summary(self, i1, i2):
        # Generate a text summary of the hyperparameters with the given indices
        # for use in naming output files.
        p1_key = self.param1.key
        p1_label = self.param1.labels[i1]
        p2_key = self.param2.key
        p2_label = self.param2.labels[i2]
        return f'{p1_key}_{p1_label}_{p2_key}_{p2_label}'

    # TODO: using the prefix "iter" is more traditional.
    def enumerate_sample_summaries(self):
        for i1 in self.param1.sample_points:
            for i2 in self.param2.sample_points:
                yield self.summary(i1, i2)

    def enumerate_batched(self):
        # For each setting of param1, return a batch of param settings and
        # sample points sweeping all values of param2.
        params = get_default_params_numpy(SWEEP_SIZE)
        for i1 in range(SWEEP_SIZE):
            params[self.param1.key] = self.param1.values[i1]
            samples = []
            for i2 in range(SWEEP_SIZE):
                params[self.param2.key][i2] = self.param2.values[i2]
                if all((i1 in self.param1.sample_points,
                        i2 in self.param2.sample_points)):
                    samples.append(i2)
            yield (params, i1, samples)

    def enumerate(self):
        # Enumerate all combinations of settings for both param1 and param2,
        # one at a time.
        params = get_default_params_numpy(1)
        for i1, i2 in np.ndindex(*SWEEP_SHAPE):
            params[self.param1.key] = self.param1.values[i1]
            params[self.param2.key] = self.param2.values[i2]
            sample = (i1 in self.param1.sample_points and
                      i2 in self.param2.sample_points)
            yield (params, (i1, i2), sample)


def bitstr_sweep(sweep, env_data, path, env_name):
    # Set up an environment for running these simulations.
    batch_size = SWEEP_SIZE * NUM_TRIALS
    env = make_field(batch_size)
    params = Params.field(shape=batch_size)
    inner_population = InnerPopulation(batch_size)

    # Sweep through hyperparameter settings one batch at a time.
    print('Evolving bitstrings for all parameters...')
    progress = trange(SWEEP_SIZE)
    frames = []
    for params_data, i1, samples in sweep.enumerate_batched():
        # Evolve a population for each batch of parameter settings, NUM_TRIALS
        # times in parallel.
        env.from_numpy(env_data[i1].repeat(NUM_TRIALS, axis=0))
        params.from_numpy(params_data.repeat(NUM_TRIALS))
        inner_population.evolve(env, params)

        # Copy fitness scores to the GPU and add them to the logs along with
        # the hyperparamter settings that correspond to those results.
        frames.append(pl.DataFrame({
            sweep.param1.name: params_data[sweep.param1.key].repeat(NUM_TRIALS),
            sweep.param2.name: params_data[sweep.param2.key].repeat(NUM_TRIALS),
            'Fitness': get_per_trial_scores(inner_population),
            'Environment': [env_name] * batch_size,
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
                inner_log = inner_population.get_logs(e)
                inner_log.write_parquet(sample_path / f'inner_log.parquet')

        progress.update()

    return pl.concat(frames)


def environment_sweep(sweep, path):
    # This number must be some whole multiple of OUTER_POPULATION_SIZE because
    # otherwise we'd need more sophisticated logic to score and propagate the
    # CPPN population.
    sims_per_batch = NUM_TRIALS * OUTER_POPULATION_SIZE

    outer_population = OuterPopulation(NUM_TRIALS)
    inner_population = InnerPopulation(sims_per_batch)
    params = Params.field(shape=sims_per_batch)
    evaluator = FitnessEvaluator(outer_population=outer_population)
    best_envs = make_flat(SWEEP_SHAPE)

    print('Evolving environments for all parameters...')
    progress = trange(SWEEP_SIZE * SWEEP_SIZE * OUTER_GENERATIONS)
    outer_population.randomize()
    for params_data, sweep_index, sample in sweep.enumerate():
        for og in range(OUTER_GENERATIONS):
            env = outer_population.make_environments()
            params.from_numpy(params_data)
            inner_population.evolve(env, params)
            evaluator.score_populations(inner_population, og)
            if og + 1 < OUTER_GENERATIONS:
                outer_population.propagate(og)

        # TODO: The results actually look really noisy! Occasionally we
        # get abysmal performance for a single configuration, when its
        # neighbors do fine. Perhaps we could be more clever at picking
        # environments from this large population that are more
        # reliable, and not just the ones that did best in one trial.
        env_data = env.to_numpy()
        best_trial = evaluator.get_best_trial(og)
        best_envs[sweep_index] = env_data[best_trial]

        if sample:
            sample_path = path / sweep.summary(*sweep_index)/ 'cppn'
            outer_population.get_logs().write_parquet(sample_path / 'outer_log.parquet')
            for t, e in enumerate(evaluator.get_best_per_trial(og)):
                is_best = '_best' if t == best_trial else ''
                np.save(sample_path / f'cppn_{t}{is_best}.npy', env_data[e])

        progress.update()

    return best_envs


def get_data(sweep, path, env_name):
    # If this sweep is already completed, just reload the results.
    data_filename = path / f'{env_name}.parquet'
    if data_filename.exists():
        data = pl.read_parquet(data_filename)
    else:
        # If we're evolving the environments to simulate...
        if env_name == 'cppn':
            # If we already evolved the environments, just load those from
            # disk. Otherwise, evolve a new environment for all the
            # hyperparameter settings in this sweep.
            envs_path = path / 'cppn_envs.npy'
            if envs_path.exists():
                env_data = np.load(envs_path)
            else:
                env_data = environment_sweep(sweep, path)
                np.save(envs_path, env_data)
        else:
            # Otherwise, just grab a static environment by name.
            env_data = STATIC_ENVIRONMENTS[env_name](SWEEP_SHAPE)

        # Evolve bitstrings in this environment for all the hyperparameter
        # settings in this sweep, then save the results to disk.
        data = bitstr_sweep(sweep, env_data, path, env_name)
        data.write_parquet(data_filename)
    return data


def pivot(sweep, data):
    p1_name, p2_name = sweep.param1.name, sweep.param2.name
    # Sadly, Polars' pivot method isn't compatible with Seaborn, so use Pandas.
    return data.to_pandas().pivot_table(
        index=p2_name, columns=p1_name, values='Fitness', aggfunc='mean')


def draw_sample_points(sweep):
    x_points, y_points = sweep.sample_points() + 0.5
    plt.plot(x_points, y_points, 'kx', ms=10)


def save_heatmap(sweep, path, env_name):
    # Either generate or load from disk the results of the sweep.
    data = get_data(sweep, path, env_name)

    # Render, decorate, and save the heatmap.
    sns.heatmap(pivot(sweep, data), **sweep.labels(), cmap='viridis')
    draw_sample_points(sweep)
    plt.suptitle(env_name)
    plt.tight_layout()
    plt.savefig(path / f'{env_name}.png', dpi=600)
    plt.close()


def compare_envs(sweep, path):
    data = {}
    for env_name in ['flat', 'baym', 'cppn']:
        data_filename = path / f'{env_name}.parquet'
        if data_filename.exists():
            data[env_name] = pl.read_parquet(data_filename)

    for env_name1, env_name2 in combinations(data.keys(), 2):
        pt1 = pivot(sweep, data[env_name1])
        pt2 = pivot(sweep, data[env_name2])
        delta = pt2 - pt1
        sns.heatmap(delta, **sweep.labels(), cmap='Spectral')
        draw_sample_points(sweep)
        plt.suptitle(f'{env_name2} - {env_name1}')
        plt.tight_layout()
        plt.savefig(path / f'{env_name2}_vs_{env_name1}.png', dpi=600)
        plt.close()


def main(env_name, sweep_kind):
    ti.init(ti.cuda, unrolling_limit=0)

    path = OUTPUT_PATH / f'{sweep_kind}_sweep'
    path.mkdir(exist_ok=True, parents=True)
    sweep = Sweep(sweep_kind)

    save_heatmap(sweep, path, env_name)
    compare_envs(sweep, path)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Run a selection param sweep on the chosen environment.')
    parser.add_argument(
        'env_name', type=str, choices=ALL_ENVIRONMENT_NAMES,
        help='Which environment to use for this experiment.')
    parser.add_argument(
        'sweep_kind', type=str, choices=SWEEP_KINDS,
        help='Which set of hyperparameters to sweep over.')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
