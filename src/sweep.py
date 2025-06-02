"""Code for running and visualizing hyperparameter sweeps.

The Param and Sweep classes are used to describe two-dimensional sweeps over
hyperparameters, and all the metadata needed to track sample points in
hyperparameter space to capture full logs for. The rest of this module is used
to render the results of running such sweeps on bitstrings and environments,
using the scripts in the corresponding sub-folders.
"""

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.constants import CARRYING_CAPACITY, ENV_NAMES
from src.environments.fitness import MAX_ENV_FITNESS
from src.graphics import BITSTR_PALETTE, ENV_NAME_PALETTE, FITNESS_DELTA_PALETTE
from src.bitstrings.population import get_default_params


SWEEP_SIZE = CARRYING_CAPACITY
SWEEP_SHAPE = (SWEEP_SIZE, SWEEP_SIZE)
SWEEP_KINDS = ['selection', 'ratchet']


def sweep_samples():
    """Directories where logs are saved, one for each sample point.
    """
    return {
        sweep_kind: list(Sweep(sweep_kind).iter_sample_summaries())
        for sweep_kind in SWEEP_KINDS
    }


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
        # Format param values for rendering in a Seaborn heatmap. We only show
        # every fourth label to avoid making the charts overcrowded.
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

    def iter_batched(self):
        # For each setting of param1, return a batch of param settings and
        # sample points sweeping all values of param2.
        params = get_default_params(SWEEP_SIZE)
        for i1 in range(SWEEP_SIZE):
            params[self.param1.key] = self.param1.values[i1]
            samples = []
            for i2 in range(SWEEP_SIZE):
                params[self.param2.key][i2] = self.param2.values[i2]
                if any(np.all(self.sample_points == [i1, i2], axis=1)):
                    samples.append(i2)
            yield (params, i1, samples)

    def iter(self):
        # Enumerate all combinations of settings for both param1 and param2,
        # one at a time.
        params = get_default_params(1)
        for i1, i2 in np.ndindex(*SWEEP_SHAPE):
            params[self.param1.key] = self.param1.values[i1]
            params[self.param2.key] = self.param2.values[i2]
            sample = any(np.all(self.sample_points == [i1, i2], axis=1))
            yield (params, (i1, i2), sample)


def pivot(sweep, data):
    p1_name, p2_name = sweep.param1.name, sweep.param2.name
    # Sadly, Polars' pivot method isn't compatible with Seaborn, so use Pandas.
    return data.to_pandas().pivot_table(
        index=p2_name, columns=p1_name, values='Fitness', aggfunc='mean')


def draw_sample_points(sweep):
    x_points, y_points = sweep.sample_points.T + 0.5
    plt.plot(x_points, y_points, 'kx', ms=10)


def render_one(sweep, path, pivot_tables, env_name, mark_samples):
    plt.figure(figsize=(2.667, 2.667))
    sns.heatmap(pivot_tables[env_name], **sweep.labels(), cbar=False,
                vmin=0, vmax=MAX_ENV_FITNESS, cmap=BITSTR_PALETTE)
    if mark_samples:
        draw_sample_points(sweep)
    plt.tight_layout()
    plt.savefig(path / f'{env_name}.png', dpi=150)
    plt.close()


def render_comparisons(sweep, path, pivot_tables, mark_samples):
    def render_delta(delta, filename):
        plt.figure(figsize=(2.667, 2.667))
        sns.heatmap(delta / MAX_ENV_FITNESS, **sweep.labels(),
                    vmin=-0.5, vmax=0.5, center=0,
                    cmap=FITNESS_DELTA_PALETTE, cbar=False)
        if mark_samples:
            draw_sample_points(sweep)
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


def main(path, sweep_kind, mark_samples):
    sweep = Sweep(sweep_kind)

    frames = []
    pivot_tables = {}
    for env_name in ENV_NAMES:
        env_data = pl.read_parquet(
            path / f'{env_name}.parquet'
        ).with_columns(
            Environment=pl.lit(env_name)
        )
        frames.append(env_data)
        pivot_tables[env_name] = pivot(sweep, env_data)
        render_one(sweep, path, pivot_tables, env_name, mark_samples)
    all_data = pl.concat(frames)

    render_comparisons(sweep, path, pivot_tables, mark_samples)
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
    parser.add_argument(
        '--mark_samples', '-s', type=BooleanOptionalAction, default=True,
        help='Show where in the sweep full-logs were sampled')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
