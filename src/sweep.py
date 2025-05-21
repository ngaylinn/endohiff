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

from .constants import CARRYING_CAPACITY, ENV_NAMES
from .graphics import BITSTRING_PALETTE, ENV_NAME_PALETTE, FITNESS_DELTA_PALETTE

# TODO: Load this from elsewhere.
MAX_OUTER_FITNESS = 449

# from compare_experiments import HUE_ORDER
# from constants import (
#     CARRYING_CAPACITY, ENV_NAMES, NUM_TRIALS, OUTER_GENERATIONS, OUTER_POPULATION_SIZE,
#     OUTPUT_PATH)
# from environments import (
#     ALL_ENVIRONMENT_NAMES, STATIC_ENVIRONMENTS, make_env_field, make_flat)
# from inner_population import InnerPopulation, get_default_params_numpy, Params
# from outer_population import OuterPopulation
# from outer_fitness import MAX_OUTER_FITNESS, FitnessEvaluator, get_per_trial_scores

SWEEP_SIZE = CARRYING_CAPACITY
SWEEP_SHAPE = (SWEEP_SIZE, SWEEP_SIZE)
SWEEP_KINDS = ['selection', 'ratchet']

def all_sweep_sample_dirs():
    summaries = []
    for sweep_kind in SWEEP_KINDS:
        summaries.extend(
            [f'{sweep_kind}_sweep/{summaries}' for summaries in
             Sweep(sweep_kind).iter_sample_summaries()])
    return summaries


class Param:
    """Represents a single hyperparameter we might sweep over.
    """
    def __init__(self, key, values):
        self.key = key
        self.name = key.title().replace('_', ' ')
        self.values = values
        if values.dtype.kind == 'f':
            self.labels = [f'{val:0.3f}' for val in self.values]
        else:
            self.labels = self.values


class Sweep:
    """Represents a 2D sweep over a pair of hyperparameters.
    """
    def __init__(self, sweep_kind):
        # The sweep configurations used by this project:
        if sweep_kind == 'selection':
            self.param1 = Param(
                'mortality_rate',
                np.linspace(0, 1, SWEEP_SIZE))
            self.param2 = Param(
                'tournament_size',
                np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1)
            self.sample_points = np.array(
                [[3, 5], [1, 10], [0, 18], [8, 18], [12, 10], [16, 18]])
        elif sweep_kind == 'ratchet':
            self.param1 = Param(
                'migration_rate',
                np.linspace(0, 3, SWEEP_SIZE))
            self.param2 = Param(
                'fertility_rate',
                np.arange(CARRYING_CAPACITY, dtype=np.int8) + 1)
            self.sample_points = np.array(
                [[1, 2], [1, 6], [5, 2], [5, 6], [12, 2], [12, 6]])

    def labels(self):
        # Format param values for rendering in a Seaborn heatmap
        return {
            'xticklabels': [
                label if i % 4 == 0 else ''
                for i, label in enumerate(self.param1.labels)
            ],
            'yticklabels': [
                label if i % 4 == 0 else ''
                for i, label in enumerate(self.param2.labels)
            ],
        }

    def summary(self, i1, i2):
        # Generate a text summary of the hyperparameters with the given indices
        # for use in naming output files.
        p1_key = self.param1.key
        p1_label = self.param1.labels[i1]
        p2_key = self.param2.key
        p2_label = self.param2.labels[i2]
        return f'{p1_key}_{p1_label}_{p2_key}_{p2_label}'

    def iter_sample_summaries(self):
        for i1, i2 in self.sample_points:
            yield self.summary(i1, i2)

#    def iter_batched(self):
#        # For each setting of param1, return a batch of param settings and
#        # sample points sweeping all values of param2.
#        params = get_default_params_numpy(SWEEP_SIZE)
#        for i1 in range(SWEEP_SIZE):
#            params[self.param1.key] = self.param1.values[i1]
#            samples = []
#            for i2 in range(SWEEP_SIZE):
#                params[self.param2.key][i2] = self.param2.values[i2]
#                if any(np.all(self.sample_points == [i1, i2], axis=1)):
#                    samples.append(i2)
#            yield (params, i1, samples)
#
#    def iter(self):
#        # Enumerate all combinations of settings for both param1 and param2,
#        # one at a time.
#        params = get_default_params_numpy(1)
#        for i1, i2 in np.ndindex(*SWEEP_SHAPE):
#            params[self.param1.key] = self.param1.values[i1]
#            params[self.param2.key] = self.param2.values[i2]
#            sample = any(np.all(self.sample_points == [i1, i2], axis=1))
#            yield (params, (i1, i2), sample)


#def bitstr_sweep(sweep, env_data, path, env_name):
#    # Set up an environment for running these simulations.
#    batch_size = SWEEP_SIZE * NUM_TRIALS
#    env = make_env_field(batch_size)
#    params = Params.field(shape=batch_size)
#    inner_population = InnerPopulation(batch_size)
#
#    # Sweep through hyperparameter settings one batch at a time.
#    print('Evolving bitstrings for all parameters...')
#    progress = trange(SWEEP_SIZE)
#    frames = []
#    for params_data, i1, samples in sweep.iter_batched():
#        # Evolve a population for each batch of parameter settings, NUM_TRIALS
#        # times in parallel.
#        env.from_numpy(env_data[i1].repeat(NUM_TRIALS, axis=0))
#        params.from_numpy(params_data.repeat(NUM_TRIALS))
#        inner_population.evolve(env, params)
#
#        # Copy fitness scores to the GPU and add them to the logs along with
#        # the hyperparamter settings that correspond to those results.
#        frames.append(pl.DataFrame({
#            sweep.param1.name: params_data[sweep.param1.key].repeat(NUM_TRIALS),
#            sweep.param2.name: params_data[sweep.param2.key].repeat(NUM_TRIALS),
#            'Fitness': get_per_trial_scores(inner_population),
#            'Environment': [env_name] * batch_size,
#        }))
#
#        # For all the sample points in this batch...
#        for i2 in samples:
#            for t in range(NUM_TRIALS):
#                # This whole batch corresponds to one value of i1, but several
#                # values of i2, each repeated NUM_TRIALS times. So, compute the
#                # environment index from the parameter and trial indices.
#                e = i2 * NUM_TRIALS + t
#
#                # Record the environment and the full bitstring evolution log
#                # for all trials with these hyperparameter settings.
#                sample_path = (
#                    path / sweep.summary(i1, i2) / env_name / f'trial_{t}')
#                sample_path.mkdir(exist_ok=True, parents=True)
#                np.save(sample_path / 'env.npy', env_data[i1, i2])
#                inner_log = inner_population.get_logs(e)
#                inner_log.write_parquet(sample_path / f'inner_log.parquet')
#
#        progress.update()
#
#    return pl.concat(frames)
#
#
#def environment_sweep(sweep, path):
#    # This number must be some whole multiple of OUTER_POPULATION_SIZE because
#    # otherwise we'd need more sophisticated logic to score and propagate the
#    # CPPN population.
#    sims_per_batch = NUM_TRIALS * OUTER_POPULATION_SIZE
#
#    outer_population = OuterPopulation(NUM_TRIALS)
#    inner_population = InnerPopulation(sims_per_batch)
#    params = Params.field(shape=sims_per_batch)
#    evaluator = FitnessEvaluator(outer_population=outer_population)
#    best_envs = make_flat(SWEEP_SHAPE)
#
#    print('Evolving environments for all parameters...')
#    progress = trange(SWEEP_SIZE * SWEEP_SIZE * OUTER_GENERATIONS)
#    for params_data, sweep_index, sample in sweep.iter():
#        outer_population.randomize()
#        for og in range(OUTER_GENERATIONS):
#            env = outer_population.make_environments()
#            params.from_numpy(params_data.repeat(sims_per_batch))
#            inner_population.evolve(env, params)
#            evaluator.score_populations(inner_population, og)
#            if og + 1 < OUTER_GENERATIONS:
#                outer_population.propagate(og)
#            progress.update()
#
#        # TODO: The results actually look really noisy! Occasionally we
#        # get abysmal performance for a single configuration, when its
#        # neighbors do fine. Perhaps we could be more clever at picking
#        # environments from this large population that are more
#        # reliable, and not just the ones that did best in one trial.
#        env_data = env.to_numpy()
#        best_trial = evaluator.get_best_trial(og)
#        best_envs[sweep_index] = env_data[best_trial]
#
#        # TODO: This is doing something a little non-sensical. There are
#        # actually several sample points per trial, but we're just saving them
#        # all according to their trial #. Either we should save the env from
#        # the best trial in all sample conditions, or the best env for each
#        # trial.
#        if sample:
#            sample_path = path / sweep.summary(*sweep_index)/ 'cppn'
#            sample_path.mkdir(exist_ok=True, parents=True)
#            outer_population.get_logs().write_parquet(sample_path / 'outer_log.parquet')
#            for t, e in enumerate(evaluator.get_best_per_trial(og)):
#                np.save(sample_path / f'cppn_{t}.npy', env_data[e])
#
#        progress.update()
#
#    return best_envs


#def get_data(sweep, path, env_name):
#    # If this sweep is already completed, just reload the results.
#    data_filename = path / f'{env_name}.parquet'
#    if data_filename.exists():
#        data = pl.read_parquet(data_filename)
#    else:
#        # and visualization for sweeps.
#        # If we're evolving the environments to simulate...
#        if env_name == 'cppn':
#            # If we already evolved the environments, just load those from
#            # disk. Otherwise, evolve a new environment for all the
#            # hyperparameter settings in this sweep.
#            envs_path = path / 'cppn_envs.npy'
#            if envs_path.exists():
#                env_data = np.load(envs_path)
#            else:
#                env_data = environment_sweep(sweep, path)
#                np.save(envs_path, env_data)
#        else:
#            # Otherwise, just grab a static environment by name.
#            env_data = STATIC_ENVIRONMENTS[env_name](SWEEP_SHAPE)
#
#        # Evolve bitstrings in this environment for all the hyperparameter
#        # settings in this sweep, then save the results to disk.
#        data = bitstr_sweep(sweep, env_data, path, env_name)
#        data.write_parquet(data_filename)
#    return data


def pivot(sweep, data):
    p1_name, p2_name = sweep.param1.name, sweep.param2.name
    # Sadly, Polars' pivot method isn't compatible with Seaborn, so use Pandas.
    return data.to_pandas().pivot_table(
        index=p2_name, columns=p1_name, values='Fitness', aggfunc='mean')


def draw_sample_points(sweep):
    x_points, y_points = sweep.sample_points.T + 0.5
    plt.plot(x_points, y_points, 'kx', ms=10)


def render_one(sweep, path, pivot_tables, env_name):
    plt.figure(figsize=(2.667, 2.667))
    sns.heatmap(pivot_tables[env_name], **sweep.labels(), cbar=False,
                vmin=0, vmax=MAX_OUTER_FITNESS, cmap=BITSTRING_PALETTE)
    #draw_sample_points(sweep)
    plt.tight_layout()
    plt.savefig(path / f'{env_name}.png', dpi=150)
    plt.close()


def render_comparisons(sweep, path, pivot_tables):
    def render_delta(delta, filename):
        plt.figure(figsize=(2.667, 2.667))
        sns.heatmap(delta / MAX_OUTER_FITNESS, **sweep.labels(),
                    vmin=-0.5, vmax=0.5, center=0,
                    cmap=FITNESS_DELTA_PALETTE, cbar=False)
        #draw_sample_points(sweep)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

    render_delta(
        pivot_tables['baym'] - pivot_tables['flat'],
        path / f'baym_vs_flat.png')

    render_delta(
        pivot_tables['cppn'] -
        np.max([pivot_tables['flat'], pivot_tables['baym']], axis=0),
        path / f'cppn_vs_both.png')


def chart_histogram(path, all_data):
    sns.displot(data=all_data, x='Fitness', hue='Environment',
                hue_order=ENV_NAMES, palette=ENV_NAME_PALETTE, legend=False,
                kind='hist', binwidth=32, fill=True, multiple='dodge',
                height=2.667, aspect=2, shrink=0.8)
    plt.xticks(np.linspace(0, 448, 8))
    plt.xticks(np.linspace(0, 448, 15), minor=True)
    plt.tight_layout()
    plt.savefig(path / f'sweep_hist.png', dpi=150)
    plt.close()


def main(path, sweep_kind):
    sweep = Sweep(sweep_kind)

    frames = []
    pivot_tables = {}
    for env_name in ENV_NAMES:
        env_data = pl.read_parquet(path / f'{env_name}.parquet')
        frames.append(env_data)
        pivot_tables[env_name] = pivot(sweep, env_data)
        render_one(sweep, path, pivot_tables, env_name)
    all_data = pl.concat(frames)

    render_comparisons(sweep, path, pivot_tables)
    chart_histogram(path, all_data)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize results from all hyperparameter sweeps.')
    parser.add_argument(
        'path', type=Path,
        help='Path containing sweep logs for all environments')
    parser.add_argument(
        'sweep_kind', type=str, choices=SWEEP_KINDS,
        help='Which kind of hyparparameter sweep (deterimes sample points).')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
