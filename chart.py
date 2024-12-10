"""Chart results from each experiment, independent of the others.
"""

import warnings

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import trange

from constants import MAX_HIFF, OUTPUT_PATH
from environment import ENVIRONMENTS


# Making ridgeplots with Seaborn generates lots of these warnings.
warnings.filterwarnings('ignore', category=UserWarning, message='Tight layout not applied')


def chart_hiff_dist(path, name, expt_data):
    """Generate a ridgeplot showing hiff distribution over generations.
    """
    expt_data = expt_data.filter(
        # Looking only at living individuals...
        (pl.col('id') > 0) &
        # Sample every ten generations...
        (pl.col('Generation') % 10 == 9)
    ).group_by(
        # For all cells across all generations...
        'Generation', 'x', 'y'
    ).agg(
        # Find the mean hiff score for all individuals in this cell.
        pl.col('hiff').mean().alias('Mean Hiff')
    )

    # Set up the ridge plot visualization.
    sns.set_theme(style='white', rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
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
    grid.figure.suptitle(f'Hiff score distribution ({name})')
    grid.figure.supylabel('Density')
    grid.figure.savefig(path / 'hiff_dist.png', dpi=600)
    plt.close()

    # Restore the default colormap so we don't alter other charts generated
    # after this one.
    plt.set_cmap('viridis')


def chart_all_results():
    """Chart results from each experiment.
    """
    num_artifacts = 2 * 2 * 1 * len(ENVIRONMENTS)
    progress = trange(num_artifacts)

    # Iterate through all the experiment conditions...
    for crossover in [True, False]:
        for migration in [True, False]:
            for env in ENVIRONMENTS.keys():
                name = env
                path = OUTPUT_PATH / f'migration_{migration}_crossover_{crossover}' / f'{env}'

                # Summarize results from a single experimental run (the best
                # one for this condition).
                expt_data = pl.read_parquet(path / 'best_trial.parquet')
                chart_hiff_dist(path, name, expt_data)
                progress.update()


if __name__ == '__main__':
    chart_all_results()
