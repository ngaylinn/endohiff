"""Visualize the performance of an inner popuation from a single experiment trial.
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
    ENVIRONMENT_SHAPE, INNER_GENERATIONS, MAX_HIFF, POP_TILE_SIZE)

ENV_PALETTE = 'mako'
POP_PALETTE = 'rocket'


def render_map_decorations(title, tile_size=1):
    """Set up and add label a figure for rendering a spatial map.

    This assumes data being rendered is tile_size * ENVIRONMENT_SHAPE and draws
    a tick mark for every 4th row / column in the environment.
    """
    plt.suptitle(title)
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
    plt.tight_layout()


def frameless_figure():
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig


def render_env_map(env_data):
    """Render a map of minimum fitness values in an environment.
    """
    frameless_figure()
    plt.imshow(env_data.transpose(), ENV_PALETTE)
    plt.clim(0, MAX_HIFF)
    # render_map_decorations('Environment Min Fitness')


def save_env_map(path, env_data, name='env_map'):
    """Summarize an environment with maps for min fitness and substr weights.
    """
    render_env_map(env_data)
    plt.savefig(path / f'{name}.png', dpi=20)
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


def render_fitness_map(inner_log):
    frameless_figure()

    pop_data = inner_log.filter(
        pl.col('Generation') == INNER_GENERATIONS - 1
    ).select('fitness').to_numpy().squeeze()

    plt.imshow(spatialize_pop_data(pop_data), POP_PALETTE)
    plt.clim(0, MAX_HIFF)
    # render_map_decorations('HIFF score map', POP_TILE_SIZE)


def save_fitness_map(path, inner_log):
    """Render a static map of final fitness scores from a population.
    """
    render_fitness_map(inner_log)
    plt.savefig(path / 'fitness_map.png', dpi=300)
    plt.close()


def save_fitness_animation(path, inner_log, gif=True):
    """Save a video of the inner population over a single experiment.
    """
    # Grab the data we need and split it by generation.
    fitness_by_generation = inner_log.select(
        'fitness'
    ).to_numpy().reshape(INNER_GENERATIONS, -1)

    fig = frameless_figure()

    # Render the first frame, and make an animation for the rest.
    image = plt.imshow(
        spatialize_pop_data(fitness_by_generation[0]), POP_PALETTE)
    plt.clim(0, MAX_HIFF)
    def animate_func(generation):
        image.set_array(spatialize_pop_data(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)

    if gif:
        anim.save(path / 'fitness_map.gif', writer='pillow', dpi=20)
    else:
        anim.save(path / 'fitness_map.mp4', writer='ffmpeg')
    plt.close()


def visualize_experiment(path, inner_log, env_data, verbose=1):
    """Generate all single trial visualizations, and save to path.
    """
    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        num_artifacts = 3
        tick_progress = trange(num_artifacts).update
    else:
        tick_progress = lambda: None

    save_fitness_animation(path, inner_log)
    tick_progress()

    save_fitness_map(path, inner_log)
    tick_progress()

    save_env_map(path, env_data)
    tick_progress()


def main(path, verbose):
    inner_log = pl.read_parquet(path / 'inner_log.parquet')
    env_data = np.load(path / 'env.npy')
    visualize_experiment(path, inner_log, env_data, verbose)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate visualizations of an inner population from single experiment trial.')
    parser.add_argument(
        'path', type=Path, help='Where to find experiment result data.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
