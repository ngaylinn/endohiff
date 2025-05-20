from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import sys

import einops
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import polars as pl

from ..constants import (
    ENVIRONMENT_SHAPE, MAX_HIFF, POP_TILE_SIZE, INNER_GENERATIONS)
from ..graphics import BITSTRING_PALETTE, frameless_figure


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
    plt.imshow(spatialize_pop_data(pop_data),
               plt.get_cmap(BITSTRING_PALETTE).with_extremes(bad='black'))
    plt.clim(0, MAX_HIFF)


def save_fitness_map(log_data, output_filename):
    """Render a static map of final fitness scores from a population.
    """
    render_fitness_map(log_data)
    plt.savefig(output_filename, dpi=20)
    plt.close()


def save_fitness_animation(log_data, output_filename):
    """Save a video of the inner population over a single experiment.
    """
    # Grab the data we need and split it by generation.
    fitness_by_generation = log_data.select(
        'fitness'
    ).to_numpy().reshape(INNER_GENERATIONS, -1)

    # Render the first frame, and make an animation for the rest.
    fig = frameless_figure()
    image = plt.imshow(
        spatialize_pop_data(fitness_by_generation[0]),
        plt.get_cmap(BITSTRING_PALETTE).with_extremes(bad='black'))
    plt.clim(0, MAX_HIFF)
    def animate_func(generation):
        image.set_array(spatialize_pop_data(fitness_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)

    # Choose an appropriate animation writer for this output file, and save.
    match output_filename.suffix:
        case '.gif':
            anim.save(output_filename, writer='pillow', dpi=20)
        case '.mp4':
            anim.save(output_filename, writer='ffmpeg')
        case _:
            pass

    plt.close()


def main(input_filename, output_filename=None, full_video=False):
    if output_filename is None:
        stem = 'fitness_map'
        suffix = '.gif' if full_video else '.png'
        output_filename = input_filename.with_stem(stem).with_suffix(suffix)

    log_data = pl.read_parquet(input_filename)
    if full_video:
        save_fitness_animation(log_data, output_filename)
    else:
        save_fitness_map(log_data, output_filename)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize the fitness of a spatial bitstring population.')
    parser.add_argument(
        'input_filename', type=Path,
        help='A bitstring evolution log to visualize (inner_log.parquet)')
    parser.add_argument(
        '--output_filename', '-o', type=Path, default=None,
        help='Where to save the image (defaults to fitness_map.png)')
    parser.add_argument(
        '--full_video', '-f', type=bool, default=False,
        action=BooleanOptionalAction,
        help='Save a full video, not just the last frame (default False)')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
