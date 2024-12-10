"""Visualize environments and populations with spatial maps.
"""

from math import sqrt

import einops
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import trange

from constants import (
    BITSTR_POWER, BITSTR_LEN, CARRYING_CAPACITY, ENVIRONMENT_SHAPE,
    INNER_GENERATIONS, MAX_HIFF, OUTPUT_PATH)
from environment import ENVIRONMENTS

POP_TILE_SIZE = int(sqrt(CARRYING_CAPACITY))


def render_map_decorations(tile_size=1):
    """Set up and add label a figure for rendering a spatial map.

    This assumes data being rendered is tile_size * ENVIRONMENT_SHAPE and draws
    a tick mark for every 4th row / column in the environment.
    """
    plt.colorbar()
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')

    # Draw a grid with lines and labels every 4 cells.
    labels = [
        np.arange(0, ENVIRONMENT_SHAPE[0] + 1, 4),
        np.arange(0, ENVIRONMENT_SHAPE[1] + 1, 4)
    ]
    # Labels and ticks are offset by 0.5 because that's how matplotlib likes to
    # align image data to the grid.
    plt.xticks(ticks=labels[0] * tile_size - 0.5, labels=labels[0])
    plt.yticks(ticks=labels[1] * tile_size - 0.5, labels=labels[1])
    plt.grid(color='k')


def render_env_map_min_fitness(name, env_data):
    """Render a map of minimum fitness values in an environment.
    """
    plt.figure(figsize=(8,4.5))
    plt.imshow(env_data['min_fitness'].transpose())
    plt.clim(0.0, MAX_HIFF)
    render_map_decorations()
    plt.suptitle(f'Environment Min Fitness ({name})')
    plt.tight_layout()


def render_env_map_weights(name, env_data):
    """Render a map of minimum fitness values in an environment.

    This function summarizes all the substring weights for a location in a
    single, densely packed visualization. For each location, there's one column
    for each length of substring, and that column is subdivided into one row
    for each substring of that length. This means the rightmost column
    (representing the one 64-bit substring) has just one row, and the leftmost
    column (representing the 32 2-bit substrings) has 32 rows.
    """
    tile_size = (BITSTR_LEN // 2)
    scaled_shape = tile_size * np.array(ENVIRONMENT_SHAPE)
    env_map = np.zeros(scaled_shape)
    bar_width = tile_size // BITSTR_POWER
    # For each location in the environment...
    for (x, y) in np.ndindex(ENVIRONMENT_SHAPE):
        # Render a tile summarizing the weights at that location.
        map_tile = np.zeros((tile_size, tile_size))
        weights = env_data['weights'][x, y]
        w = 0

        # For all the different lengths of substring...
        for p in range(BITSTR_POWER):
            # Look up the relevant weights and paint them into the map tile.
            substr_len = 2 ** (p + 1)
            substr_count = BITSTR_LEN // substr_len
            substr_weights = weights[w:w + substr_count]
            map_tile[p * bar_width:] = substr_weights.repeat(
                tile_size // substr_count)
            w += substr_count

        # Place this tile into the overall map.
        env_map[x * tile_size:(x + 1) * tile_size,
                y * tile_size:(y + 1) * tile_size] = map_tile

    # Actually draw the figure.
    plt.figure(figsize=(8,4.5))
    plt.imshow(env_map.transpose())
    plt.clim(0.0, 1.0)
    render_map_decorations(tile_size)
    plt.suptitle(f'Environment Substring Weights ({name})')
    plt.tight_layout()


def save_env_map(path, name, env_data):
    """Summarize an environment with maps for min fitness and substr weights.
    """
    render_env_map_min_fitness(name, env_data)
    plt.savefig(path / 'env_map_fitness.png', dpi=600)
    plt.close()

    render_env_map_weights(name, env_data)
    plt.savefig(path / 'env_map_weights.png', dpi=600)
    plt.close()


def spatialize_pop_data(pop_data):
    """Transform raw population data from logs into a spatial layout to render.

    This produces an array where the local population at each location is
    arranged into a square tile, and all the tiles are composed into a single
    map representing the full population across all environmental locations.
    """
    ew, eh = ENVIRONMENT_SHAPE
    cw, ch = [POP_TILE_SIZE] * 2
    return einops.rearrange(
        pop_data, '(ew eh cw ch) -> (eh ch) (ew cw)',
        ew=ew, eh=eh, cw=cw, ch=ch)


def render_pop_map(pop_map):
    """Render a map of population fitness / hiff scores to the current figure.

    This is shared by code below to ensure the same color map is used
    everywhere, including rendering unpopulated spaces in black.
    """
    image = plt.imshow(pop_map, plt.get_cmap().with_extremes(under='black'))
    plt.clim(1, MAX_HIFF)
    return image


def save_hiff_map(path, name, expt_data):
    """Render a static map of final hiff scores from a population.
    """
    plt.figure(figsize=(8,4.5))
    pop_data = expt_data.select('hiff').to_numpy().flatten()
    render_pop_map(spatialize_pop_data(pop_data))
    render_map_decorations(POP_TILE_SIZE)
    plt.suptitle(f'HIFF score map ({name})')
    plt.tight_layout()
    plt.savefig(path / 'hiff_map.png', dpi=600)
    plt.close()


def save_hiff_animation(path, expt_data, gif=False):
    """Save a video of the inner population over a single experiment.
    """
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
    image = render_pop_map(spatialize_pop_data(fitness_by_generation[0]))
    def animate_func(generation):
        image.set_array(spatialize_pop_data(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)

    if gif:
        anim.save(path / 'hiff_map.gif', writer='pillow', dpi=20)
    else:
        anim.save(path / 'hiff_map.mp4', writer='ffmpeg')


def save_all_results():
    """Render all visualizations for all primary experiments.
    """
    num_artifacts = 2 * 2 * 3 * len(ENVIRONMENTS)
    progress = trange(num_artifacts)
    for crossover in [True, False]:
        for migration in [True, False]:
            for env in ENVIRONMENTS.keys():
                name = env
                path = OUTPUT_PATH / f'migration_{migration}_crossover_{crossover}' / f'{env}'

                # Save an animation of fitness over time.
                best_trial = pl.read_parquet(path / 'best_trial.parquet')
                save_hiff_animation(path, best_trial)
                progress.update()

                # Restrict to the last generation and render maps of the final fitnes
                # and HIFF scores.
                best_trial = best_trial.filter(
                    pl.col('Generation') == INNER_GENERATIONS - 1
                )

                save_hiff_map(path, name, best_trial)
                progress.update()

                # Load and render the environment where this experiment happened.
                env_data = np.load(path / 'env.npz')

                save_env_map(path, name, env_data)
                progress.update()


if __name__ == '__main__':
    save_all_results()
