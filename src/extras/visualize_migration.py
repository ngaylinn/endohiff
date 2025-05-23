"""Visualize how bitstrings move with different migration_rates.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import CARRYING_CAPACITY


def main(path):
    path.mkdir(exist_ok=True, parents=True)

    for migration_rate in [0.2, 0.5, 1.0]:
        np.random.seed(42)
        sns.relplot(
            x=migration_rate * np.random.randn(CARRYING_CAPACITY),
            y=migration_rate * np.random.randn(CARRYING_CAPACITY),
            kind='scatter'
        )
        plt.xticks([-1.5, -0.5, 0.5, 1.5], labels=[])
        plt.xticks([-1, 0, 1], labels=['-1', '0', '+1'], minor=True)
        plt.yticks([-1.5, -0.5, 0.5, 1.5], labels=[])
        plt.yticks([-1, 0, 1], labels=['-1', '0', '+1'], minor=True)
        plt.grid()
        plt.tight_layout()
        plt.savefig(path / f'migration_{migration_rate:0.1f}.png')

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize bitstring migration at a few rates.')
    parser.add_argument(
        'path', type=Path,
        help='Where to save visualizations of bistring migration')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
