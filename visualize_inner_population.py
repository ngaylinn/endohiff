"""Visualize the performance of an inner popuation from a single experiment trial.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import warnings

import einops
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import (
    BITSTR_LEN, ENVIRONMENT_SHAPE, INNER_GENERATIONS, MAX_HIFF, POP_TILE_SIZE)


# Making ridgeplots with Seaborn generates lots of these warnings.
warnings.filterwarnings('ignore', category=UserWarning, message='Tight layout not applied')


def chart_fitness_dist(path, inner_log):
    """Generate a ridgeplot showing fitness distribution over generations.
    """
    num_ridges = 10
    ridge_gap = INNER_GENERATIONS // num_ridges
    sample_generations = set(
        range(ridge_gap - 1, INNER_GENERATIONS, ridge_gap))
    inner_log = inner_log.filter(
        # Looking only at living individuals...
        pl.col('alive') &
        # Sample every ten generations...
        (pl.col('Generation').is_in(sample_generations))
    ).group_by(
        # For all cells across all generations...
        'Generation', 'x', 'y'
    ).agg(
        # Find the mean fitness score for all individuals in this cell.
        pl.col('fitness').mean().alias('Mean Fitness')
    ).drop('x', 'y')

    # Since we only count living individuals, we might have no data for some
    # generations if the population went extinct. Make sure we can still
    # generate a chart by filling in data for missing generations.
    missing_generations = (
        sample_generations - set(inner_log['Generation'].unique()))
    inner_log = pl.concat((inner_log, pl.DataFrame({
        'Generation': sorted(missing_generations),
        'Mean Fitness': np.float32(0.0)
    })))

    # Add a column indicating the overall average fitness score so we can color
    # each ridge to match the performance at that time step.
    inner_log = inner_log.join(
        inner_log.group_by(
            'Generation'
        ).agg(
            pl.col('Mean Fitness').mean().alias('color')
        ),
        on='Generation',
        how='inner',
        coalesce=True
    )

    # Set up the ridge plot visualization.
    sns.set_theme(style='white', rc={"axes.facecolor": (0, 0, 0, 0)})
    grid = sns.FacetGrid(
        inner_log, row='Generation', hue='Generation',
        aspect=15, height=0.5, xlim=(0, MAX_HIFF))

    # Plot the mean fitness score for each sample generation and label it with the
    # generation number.
    def plot_ridge(x, color, label):
        ax = plt.gca()
        color = plt.cm.viridis(x.mean() / MAX_HIFF)
        sns.kdeplot(x=x, color=color, fill=True, warn_singular=False)
        ax.text(0, 0.2, f'Gen {label}', ha='left', va='center',
                transform=ax.transAxes)
    grid.map(plot_ridge, 'Mean Fitness')

    # Apply styling and save results.
    grid.refline(y=0, linestyle='-', clip_on=False)
    grid.figure.subplots_adjust(hspace=-0.25)
    grid.set_titles('')
    grid.set(yticks=[], ylabel='')
    grid.despine(bottom=True, left=True)
    grid.figure.suptitle(f'Fitness score distribution')
    grid.figure.supylabel('Density')
    grid.figure.savefig(path / 'inner_fitness.png', dpi=600)
    plt.close()

    # Restore the default colormap so we don't alter other charts generated
    # after this one.
    plt.set_cmap('viridis')


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


def render_env_map(env_data):
    """Render a map of minimum fitness values in an environment.
    """
    plt.figure(figsize=(8, 4.5))
    plt.imshow(env_data.transpose())
    plt.clim(0, MAX_HIFF)
    render_map_decorations()
    plt.suptitle(f'Environment Min Fitness')
    plt.tight_layout()


def save_env_map(path, env_data, name='env_map'):
    """Summarize an environment with maps for min fitness and substr weights.
    """
    render_env_map(env_data)
    plt.savefig(path / f'{name}.png', dpi=600)
    plt.close()


def get_masked_column_data(data, column):
    """Select a column of experiment data, with dead individuals masked out.
    """
    return np.ma.masked_array(
        data.select(column).to_numpy().squeeze(),
        ~data.select('alive').to_numpy().squeeze())


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
    plt.figure(figsize=(8, 4.5))
    pop_data = get_masked_column_data(inner_log, 'fitness')
    plt.imshow(spatialize_pop_data(pop_data),
               plt.get_cmap().with_extremes(bad='black'))
    plt.clim(0, MAX_HIFF)
    render_map_decorations(POP_TILE_SIZE)
    plt.suptitle(f'HIFF score map')
    plt.tight_layout()


def save_fitness_map(path, inner_log):
    """Render a static map of final fitness scores from a population.
    """
    render_fitness_map(inner_log)
    plt.savefig(path / 'fitness_map.png', dpi=600)
    plt.close()


def render_one_frac_map(inner_log):
    plt.figure(figsize=(8, 4.5))
    pop_data = get_masked_column_data(inner_log, 'one_count') / BITSTR_LEN
    plt.imshow(spatialize_pop_data(pop_data),
               cmap=plt.get_cmap('Spectral').with_extremes(bad='black'))
    plt.clim(0.0, 1.0)
    render_map_decorations(POP_TILE_SIZE)
    plt.suptitle(f'Bit ratio map')
    plt.tight_layout()


def save_one_frac_map(path, inner_log):
    """Render a static map of a population's ratio of 1s to 0s.
    """
    render_one_frac_map(inner_log)
    plt.savefig(path / 'one_frac_map.png', dpi=600)
    plt.close()


def save_fitness_animation(path, inner_log, gif=True):
    """Save a video of the inner population over a single experiment.
    """
    # Grab the data we need and split it by generation.
    fitness_by_generation = get_masked_column_data(
        inner_log, 'fitness'
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
        anim.save(path / 'fitness_map.gif', writer='pillow', dpi=20)
    else:
        anim.save(path / 'fitness_map.mp4', writer='ffmpeg')
    plt.close()


def save_one_frac_animation(path, inner_log, gif=True):
    """Save a video of a population's ratio of 1s to 0s over one experiment.
    """
    # Grab the data we need and split it by generation.
    one_frac_by_generation = get_masked_column_data(
        inner_log, 'one_count'
    ).reshape(INNER_GENERATIONS, -1) / BITSTR_LEN

    # Set up a figure with no decorations or padding.
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Render the first frame, and make an animation for the rest.
    image = plt.imshow(spatialize_pop_data(one_frac_by_generation[0]),
                       plt.get_cmap('Spectral').with_extremes(bad='black'))
    plt.clim(0.0, 1.0)
    def animate_func(generation):
        image.set_array(spatialize_pop_data(one_frac_by_generation[generation]))
        return image
    anim = FuncAnimation(fig, animate_func, INNER_GENERATIONS, interval=100)

    if gif:
        anim.save(path / 'one_frac_map.gif', writer='pillow', dpi=20)
    else:
        anim.save(path / 'one_frac_map.mp4', writer='ffmpeg')
    plt.close()


def visualize_experiment(path, inner_log, env_data, verbose=1):
    """Generate all single trial visualizations, and save to path.
    """
    # Maybe show a progress bar as we generate files.
    if verbose > 0:
        num_artifacts = 6
        tick_progress = trange(num_artifacts).update
    else:
        tick_progress = lambda: None

    save_one_frac_animation(path, inner_log)
    tick_progress()

    save_fitness_animation(path, inner_log)
    tick_progress()

    chart_fitness_dist(path, inner_log)
    tick_progress()

    # Restrict to the last generation and render still maps of the final state.
    inner_log = inner_log.filter(
        pl.col('Generation') == INNER_GENERATIONS - 1
    )

    save_one_frac_map(path, inner_log)
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
