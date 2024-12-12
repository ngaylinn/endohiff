"""Visualize environments and populations with spatial maps.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import einops
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import trange

from constants import (
    BITSTR_POWER, BITSTR_LEN, ENVIRONMENT_SHAPE, INNER_GENERATIONS, MAX_HIFF,
    POP_TILE_SIZE, OUTPUT_PATH)
from environment import ENVIRONMENTS


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


def render_env_map_min_fitness(env_data):
    """Render a map of minimum fitness values in an environment.
    """
    plt.figure(figsize=(8, 4.5))
    plt.imshow(env_data['min_fitness'].transpose())
    plt.clim(0, MAX_HIFF)
    render_map_decorations()
    plt.suptitle(f'Environment Min Fitness')
    plt.tight_layout()


def render_env_map_weights(env_data):
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
    plt.figure(figsize=(8, 4.5))
    plt.imshow(env_map.transpose())
    plt.clim(0.0, 1.0)
    render_map_decorations(tile_size)
    plt.suptitle(f'Environment Substring Weights')
    plt.tight_layout()


def save_env_map(path, env_data):
    """Summarize an environment with maps for min fitness and substr weights.
    """
    render_env_map_min_fitness(env_data)
    plt.savefig(path / 'env_map_fitness.png', dpi=600)
    plt.close()

    render_env_map_weights(env_data)
    plt.savefig(path / 'env_map_weights.png', dpi=600)
    plt.close()


def get_masked_column_data(data, column):
    """Select a column of experiment data, with dead individuals masked out.
    """
    return np.ma.masked_array(
        data.select(column).to_numpy().squeeze(),
        data.select('id').to_numpy().squeeze() == 0)


def get_one_frac(arr):
    """Find the ratio of 1s to 0s for each bit string (int) in an array.
    """
    one_counts = np.zeros(len(arr))
    for b in range(BITSTR_LEN):
        one_counts += (arr >> b) & 1
    return one_counts / 64


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


def save_hiff_map(path, expt_data):
    """Render a static map of final hiff scores from a population.
    """
    plt.figure(figsize=(8, 4.5))
    pop_data = get_masked_column_data(expt_data, 'hiff')
    plt.imshow(spatialize_pop_data(pop_data),
               plt.get_cmap().with_extremes(bad='black'))
    plt.clim(0, MAX_HIFF)
    render_map_decorations(POP_TILE_SIZE)
    plt.suptitle(f'HIFF score map')
    plt.tight_layout()
    plt.savefig(path / 'hiff_map.png', dpi=600)
    plt.close()


def save_one_frac_map(path, expt_data):
    """Render a static map of a population's ratio of 1s to 0s.
    """
    plt.figure(figsize=(8, 4.5))
    pop_data = get_masked_column_data(expt_data, 'one_frac')
    plt.imshow(spatialize_pop_data(pop_data),
               cmap=plt.get_cmap('Spectral').with_extremes(bad='black'))
    plt.clim(0.0, 1.0)
    render_map_decorations(POP_TILE_SIZE)
    plt.suptitle(f'Bit ratio map')
    plt.tight_layout()
    plt.savefig(path / 'one_frac_map.png', dpi=600)
    plt.close()


def save_hiff_animation(path, expt_data, gif=True):
    """Save a video of the inner population over a single experiment.
    """
    # Grab the data we need and split it by generation.
    fitness_by_generation = get_masked_column_data(
        expt_data, 'hiff'
    ).reshape(INNER_GENERATIONS, -1)

    # Set up a figure with no decorations or padding.
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Render the first frame, and make an animation for the rest.
    image = plt.imshow(spatialize_pop_data(fitness_by_generation[0]),
                       plt.get_cmap().with_extremes(bad='black'))
    plt.clim(0, MAX_HIFF)
    def animate_func(generation):
        image.set_array(spatialize_pop_data(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)

    if gif:
        anim.save(path / 'hiff_map.gif', writer='pillow', dpi=20)
    else:
        anim.save(path / 'hiff_map.mp4', writer='ffmpeg')
    plt.close()


def save_one_frac_animation(path, expt_data, gif=True):
    """Save a video of a population's ratio of 1s to 0s over one experiment.
    """
    # Grab the data we need and split it by generation.
    fitness_by_generation = get_masked_column_data(
        expt_data, 'one_frac'
    ).reshape(INNER_GENERATIONS, -1)

    # Set up a figure with no decorations or padding.
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Render the first frame, and make an animation for the rest.
    image = plt.imshow(spatialize_pop_data(fitness_by_generation[0]),
                       plt.get_cmap('Spectral').with_extremes(bad='black'))
    plt.clim(0.0, 1.0)
    def animate_func(generation):
        image.set_array(spatialize_pop_data(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)

    if gif:
        anim.save(path / 'one_frac_map.gif', writer='pillow', dpi=20)
    else:
        anim.save(path / 'one_frac_map.mp4', writer='ffmpeg')
    plt.close()


def main(best_trial_file, env_file, path, verbose):
    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        num_artifacts = 5
        tick_progress = trange(num_artifacts).update
    else:
        tick_progress = lambda: None

    # Save animations of the full evolutionary experiment.
    best_trial_data = pl.read_parquet(best_trial_file)
    best_trial_data = best_trial_data.with_columns(
        one_frac=get_one_frac(best_trial_data['bitstr'].to_numpy())
    )
    save_hiff_animation(path, best_trial_data)
    tick_progress()
    save_one_frac_animation(path, best_trial_data)
    tick_progress()

    # Restrict to the last generation and render still maps of the final state.
    best_trial = best_trial_data.filter(
        pl.col('Generation') == INNER_GENERATIONS - 1
    )
    save_hiff_map(path, best_trial)
    tick_progress()
    save_one_frac_map(path, best_trial)
    tick_progress()

    # Render visualizations of the environment for this experiment.
    env_data = np.load(env_file)
    save_env_map(path, env_data)
    tick_progress()

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Render visualizations of single experiment.')
    parser.add_argument(
        'path', type=Path, help='Where to find experiment result data.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    # Verify that the path is valid and the files we need exist.
    args['best_trial_file'] = args['path'] / 'best_trial.parquet'
    args['env_file'] = args['path'] / 'env.npz'
    if not args['best_trial_file'].exists() or not args['env_file'].exists():
        raise FileNotFoundError('Experiment result data not found.')

    # Actually render these results.
    sys.exit(main(**args))
