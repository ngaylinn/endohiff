"""Chart results from each experiment, independent of the others.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from constants import INNER_GENERATIONS, MAX_HIFF


# Making ridgeplots with Seaborn generates lots of these warnings.
warnings.filterwarnings('ignore', category=UserWarning, message='Tight layout not applied')


def chart_hiff_dist(path, expt_data):
    """Generate a ridgeplot showing hiff distribution over generations.
    """
    num_ridges = 10
    ridge_gap = INNER_GENERATIONS // num_ridges
    expt_data = expt_data.filter(
        # Looking only at living individuals...
        (pl.col('id') > 0) &
        # Sample every ten generations...
        (pl.col('Generation') % ridge_gap == ridge_gap - 1)
    ).group_by(
        # For all cells across all generations...
        'Generation', 'x', 'y'
    ).agg(
        # Find the mean hiff score for all individuals in this cell.
        pl.col('hiff').mean().alias('Mean Hiff')
    )

    # Set up the ridge plot visualization.
    sns.set_theme(style='white', rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(num_ridges, rot=-.25, light=.7)
    grid = sns.FacetGrid(
        expt_data, row='Generation', hue='Generation',
        aspect=15, height=0.5, palette=pal, xlim=(0, MAX_HIFF))

    # Plot the mean hiff score for each sample generation and label it with the
    # generation number.
    grid.map(sns.kdeplot, 'Mean Hiff', bw_adjust=1.5,
             clip_on=True, fill=True, alpha=1.0)
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, f'Gen {label}', ha='left', va='center',
                transform=ax.transAxes)
    grid.map(label, 'Mean Hiff')

    # Apply styling and save results.
    grid.refline(y=0, linestyle='-', clip_on=False)
    grid.figure.subplots_adjust(hspace=-0.25)
    grid.set_titles('')
    grid.set(yticks=[], ylabel='')
    grid.despine(bottom=True, left=True)
    grid.figure.suptitle(f'Hiff score distribution')
    grid.figure.supylabel('Density')
    grid.figure.savefig(path / 'hiff_dist.png', dpi=600)
    plt.close()

    # Restore the default colormap so we don't alter other charts generated
    # after this one.
    plt.set_cmap('viridis')


def main(best_trial_file, path, verbose):
    # Chart the hiff score distribution over time.
    best_trial_data = pl.read_parquet(best_trial_file)
    chart_hiff_dist(path, best_trial_data)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Chart results from a single experiment.')
    parser.add_argument(
        'path', type=Path, help='Where to find experiment result data.')
    # Unused. Just here for consistency with the other scripts.
    parser.add_argument(
        '-v', '--verbose', type=int, default=1,
        help='Verbosity level (1 is default, 0 for no output)')
    args = vars(parser.parse_args())

    # Verify that the path is valid and the file we need exists.
    args['best_trial_file'] = args['path'] / 'best_trial.parquet'
    if not args['best_trial_file'].exists():
        raise FileNotFoundError('Experiment result data not found.')

    # Actually render these results.
    sys.exit(main(**args))
