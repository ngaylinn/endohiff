from math import sqrt
from pathlib import Path

import einops
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import trange

from constants import (
    BITSTR_POWER, BITSTR_LEN, CARRYING_CAPACITY, ENVIRONMENT_SHAPE,
    INNER_GENERATIONS, NUM_WEIGHTS, MAX_HIFF, MIN_HIFF, OUTPUT_PATH)
from run_experiments import CONDITION_NAMES


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
    # TODO: How might we show the different weights across substrings of the
    # same size? This is a harder visualization challenge.
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


def save_env_map(path, name, env_data):
    render_env_map(env_data)
    plt.suptitle(f'Environment map ({name})')
    plt.tight_layout()
    plt.savefig(path / 'env_map.png', dpi=600)
    plt.close()


def make_pop_map(pop_data):
    # Arrange the population data spatially.
    ew, eh = ENVIRONMENT_SHAPE
    cw, ch = [int(sqrt(CARRYING_CAPACITY))] * 2
    return einops.rearrange(
        pop_data, '(ew eh cw ch) -> (eh ch) (ew cw)',
        ew=ew, eh=eh, cw=cw, ch=ch)

def render_pop_map(pop_map):
    # Render the population data to a figure, with a color scale proportional
    # to the MAX_HIFF score and dead cells rendered in black.
    image = plt.imshow(pop_map, plt.get_cmap().with_extremes(under='black'))
    plt.clim(1, MAX_HIFF)
    return image


def save_fitness_map(path, name, expt_data):
    pop_data = expt_data.select('fitness').to_numpy().flatten()
    plt.figure(figsize=(8, 4.5))
    render_pop_map(make_pop_map(pop_data))
    plt.suptitle(f'Fitness map ({name})')
    plt.colorbar()
    plt.savefig(path / 'fitness_map.png', dpi=600)
    plt.close()


def save_hiff_map(path, name, expt_data):
    pop_data = expt_data.select('hiff').to_numpy().flatten()
    plt.figure(figsize=(8, 4.5))
    render_pop_map(make_pop_map(pop_data))
    plt.suptitle(f'HIFF score map ({name})')
    plt.colorbar()
    plt.savefig(path / 'hiff_map.png', dpi=600)
    plt.close()

def save_fitness_animation(path, expt_data):
    # Grab the data we need and split it by generation.
    fitness_by_generation = expt_data.select(
        'fitness'
    ).to_numpy().reshape(INNER_GENERATIONS, -1)

    # Set up a figure with no decorations or padding.
    fig = plt.figure(frameon=False, figsize=(8, 4.5))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Render the first frame, and make an animation for the rest.
    image = render_pop_map(make_pop_map(fitness_by_generation[0]))
    def animate_func(generation):
        image.set_array(make_pop_map(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)
    anim.save(path / 'fitness_map.mp4', writer='ffmpeg')


def save_all_results():
    num_artifacts = 4 * len(CONDITION_NAMES)
    progress = trange(num_artifacts)
    for name in CONDITION_NAMES:
        path = OUTPUT_PATH / name
        expt_data = pl.read_parquet(path / 'inner_log.parquet')

        # Save an animation of fitness over time.
        save_fitness_animation(path, expt_data)
        progress.update()

        # Restrict to the last generation and render maps of the final fitnes
        # and HIFF scores.
        expt_data = expt_data.filter(
            pl.col('generation') == INNER_GENERATIONS - 1
        )

        save_fitness_map(path, name, expt_data)
        progress.update()

        save_hiff_map(path, name, expt_data)
        progress.update()

        # Load and render the environment where this experiment happened.
        env_data = np.load(path / 'env.npz')
        save_env_map(path, name, env_data)
        progress.update()


if __name__ == '__main__':
    save_all_results()
