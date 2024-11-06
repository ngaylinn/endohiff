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


BORDER_COLOR = '#007155'  # UVM Green


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
    plt.figure(figsize=(6,8))
    ax = plt.subplot(2, 1, 1)
    ax.set(title='Min Fitness')
    plt.imshow(env_data['min_fitness'].transpose())
    plt.clim(0.0, MAX_HIFF)
    plt.colorbar()
    plt.xticks(np.arange(-0.5, ENVIRONMENT_SHAPE[0]), labels=[])
    plt.yticks(np.arange(-0.5, ENVIRONMENT_SHAPE[1]), labels=[])
    plt.grid(color=BORDER_COLOR)

    tile_size = (BITSTR_LEN // 2)
    scaled_shape = tile_size * np.array(ENVIRONMENT_SHAPE)
    env_map = np.zeros(scaled_shape)
    bar_width = tile_size // BITSTR_POWER
    for (x, y) in np.ndindex(ENVIRONMENT_SHAPE):
        w = 0
        map_tile = env_map[x * tile_size:(x + 1) * tile_size,
                           y * tile_size:(y + 1) * tile_size]
        weights = env_data['weights'][x, y]
        for p in range(BITSTR_POWER):
            substr_len = 2 ** (p + 1)
            substr_count = BITSTR_LEN // substr_len
            substr_weights = weights[w:w + substr_count]
            map_tile[p * bar_width:] = substr_weights.repeat(
                tile_size // substr_count)
            w += substr_count
        env_map[x * tile_size:(x + 1) * tile_size,
                y * tile_size:(y + 1) * tile_size] = map_tile

    ax = plt.subplot(2, 1, 2)
    ax.set(title='Substring Weights')
    plt.imshow(env_map.transpose())
    plt.clim(0.0, 1.0)
    plt.colorbar()
    plt.xticks(np.arange(-0.5, scaled_shape[0], tile_size), labels=[])
    plt.yticks(np.arange(-0.5, scaled_shape[1], tile_size), labels=[])
    plt.grid(color=BORDER_COLOR)


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

def render_concentration_map(path, name, avg_scores):
    """
    Renders the concentration map (average HIFF scores per cell) as a heatmap.
    Args:
        concentration_map: 2D numpy array with shape (32, 64), where each value represents the average HIFF score of a cell.
    """
    plt.figure(figsize=(8, 4.5))
    plt.imshow(avg_scores.T, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.suptitle(f'HIFF density map ({name})')
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.savefig(path / 'avg_hiff_map.png', dpi=600)
    plt.close()


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

def save_hiff_animation(path, expt_data):
    # Grab the data we need and split it by generation.
    fitness_by_generation = expt_data.select(
        'hiff'
    ).to_numpy().reshape(INNER_GENERATIONS, -1)

    # Set up a figure with no decorations or padding.
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Render the first frame, and make an animation for the rest.
    image = render_pop_map(make_pop_map(fitness_by_generation[0]))
    def animate_func(generation):
        image.set_array(make_pop_map(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)
    anim.save(path / 'hiff_map.mp4', writer='ffmpeg')

# def save_avg_hiff_map(path, name, expt_data):
#     # Extract the HIFF scores for each individual in the population.
#     hiff_scores = expt_data.select('hiff').to_numpy()
    
#     # Reshape the hiff_scores to match the grid structure (32 x 64 cells, each with 25 individuals).
#     hiff_scores_reshaped = hiff_scores.reshape(ENVIRONMENT_SHAPE[0], ENVIRONMENT_SHAPE[1], CARRYING_CAPACITY)

#     # Compute the average HIFF score for each cell (averaging across the 25 individuals in each cell).
#     avg_hiff_scores = np.mean(hiff_scores_reshaped, axis=2)
    
#     # Render the map of average HIFF scores.
#     plt.figure(figsize=(8, 4.5))
#     render_pop_map(make_pop_map(avg_hiff_scores.flatten()))  # Flatten to match the render_pop_map input
#     plt.suptitle(f'Average HIFF score map ({name})')
#     plt.colorbar()
#     plt.savefig(path / 'avg_hiff_map.png', dpi=600)
#     plt.close()

def save_avg_hiff_map(path, name, expt_data):
    # Extract the HIFF scores for each individual in the population.
    hiff_scores = expt_data.select('hiff').to_numpy()
    
    # Reshape the HIFF scores to (32, 64, 25) shape, where each cell has 25 individuals
    # TODO: change to (64, 32, 25)
    hiff_scores_reshaped = hiff_scores.reshape(ENVIRONMENT_SHAPE[0], ENVIRONMENT_SHAPE[1], CARRYING_CAPACITY)

    # Get the aggregated average HIFF scores for each cell
    avg_hiff_scores = np.mean(hiff_scores_reshaped, axis = 2)

    # Render the concentration map of average HIFF scores
    render_concentration_map(path, name, avg_hiff_scores)


def save_all_results():
    num_artifacts = 4 * len(CONDITION_NAMES)
    progress = trange(num_artifacts)
    for name in CONDITION_NAMES:
        path = OUTPUT_PATH / name
        expt_data = pl.read_parquet(path / 'inner_log.parquet')

        # Save an animation of fitness over time.
        save_hiff_animation(path, expt_data)
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

        save_avg_hiff_map(path, name, expt_data)
        progress.update()

        # Load and render the environment where this experiment happened.
        env_data = np.load(path / 'env.npz')
        save_env_map(path, name, env_data)
        progress.update()


if __name__ == '__main__':
    save_all_results()
