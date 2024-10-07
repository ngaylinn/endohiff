from math import sqrt
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import trange

from constants import (
    BITSTR_POWER, BITSTR_LEN, CARRYING_CAPACITY, ENVIRONMENT_SHAPE,
    NUM_GENERATIONS, NUM_WEIGHTS, MAX_HIFF, MIN_HIFF)


def compute_substr_scores():
    substr_scores = np.zeros(NUM_WEIGHTS)
    w = 0
    for p in range(BITSTR_POWER):
        substr_len = 2**(1 + p)
        for s in range(BITSTR_LEN // substr_len):
            substr_scores[w] = substr_len
            w += 1
    return substr_scores

SUBSTR_SCORES = compute_substr_scores()


def render_env_map(env_data):
    # Compute 'survivability' for each cell. This is a metric based on the
    # minimum fitness required to occupy this cell and the maximum possible
    # score achievable with the given weights.
    # TODO: It might be better to compute what percent of randomly generated
    # bit strings could survive here, rather than basing this on the range of
    # possible fitness scores.
    max_scores = (env_data['weights'] * SUBSTR_SCORES).sum(axis=2)
    min_scores = env_data['min_fitness']
    survivability = (max_scores - min_scores) / (MAX_HIFF - MIN_HIFF)

    # Set up a figure and render "survivability" at the top.
    plt.figure(figsize=(4, 12))
    ax = plt.subplot(BITSTR_POWER + 1, 1, 1)
    ax.set(title='Survivability')
    plt.imshow(survivability.transpose())
    plt.clim(0.0, 1.0)
    plt.colorbar()

    # For all the different substring lengths, render the average weight given
    # to substrings of that size in their own plots.
    w = 0
    for p in range(BITSTR_POWER):
        substr_len = 2 ** (p + 1)
        substr_count = BITSTR_LEN // substr_len
        substr_weights = env_data['weights'][:, :, w:w + substr_count]
        avg_weights = substr_weights.mean(axis=2)
        ax = plt.subplot(BITSTR_POWER + 1, 1, 2 + p)
        ax.set(title=f'Average weight for {substr_len}-bit substrings')
        plt.imshow(avg_weights.transpose())
        plt.clim(0.0, 1.0)
        plt.colorbar()
        w += substr_count
    plt.tight_layout()


def save_env_map(name, env_data):
    render_env_map(env_data)
    plt.suptitle(f'Environment map ({name})')
    plt.savefig(f'output/{name}/env_map.png', dpi=600)
    plt.close()


def render_pop_map(pop_data):
    # Arrange the population data spatially.
    pop_data = pop_data.to_numpy().flatten()
    ew, eh = ENVIRONMENT_SHAPE
    cw, ch = [int(sqrt(CARRYING_CAPACITY))] * 2
    pop_map = einops.rearrange(
        pop_data, '(ew eh cw ch) -> (eh ch) (ew cw)',
        ew=ew, eh=eh, cw=cw, ch=ch)

    # Render the data to a figure.
    plt.figure(figsize=(8, 4.5))
    plt.imshow(pop_map)
    plt.clim(0, MAX_HIFF)
    plt.colorbar()


def save_fitness_map(name, expt_data):
    render_pop_map(expt_data.select('fitness'))
    plt.suptitle(f'Fitness map ({name})')
    plt.savefig(f'output/{name}/fitness_map.png', dpi=600)
    plt.close()


def save_hiff_map(name, expt_data):
    render_pop_map(expt_data.select('hiff'))
    plt.suptitle(f'HIFF score map ({name})')
    plt.savefig(f'output/{name}/hiff_map.png', dpi=600)
    plt.close()


def save_all_results():
    history = pl.read_parquet('output/history.parquet')
    conditions = history.select('condition').unique().to_series().to_list()
    num_artifacts = 3 * len(conditions)
    progress = trange(num_artifacts)
    for name in conditions:
        Path(f'output/{name}').mkdir(exist_ok=True)

        expt_data = history.filter(
            pl.col('condition') == name
        ).filter(
            pl.col('generation') == NUM_GENERATIONS - 1
        )

        save_fitness_map(name, expt_data)
        progress.update()

        save_hiff_map(name, expt_data)
        progress.update()

        env_data = np.load(f'output/env_{name}.npz')
        save_env_map(name, env_data)
        progress.update()


if __name__ == '__main__':
    save_all_results()
