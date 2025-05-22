"""Visualize how bitstrings move with a particular migration_rate.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import CARRYING_CAPACITY


def main(output_file, migration_rate):
    output_file.parent.mkdir(exist_ok=True, parents=True)

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
    plt.savefig(output_file)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize a bitstring migration rate.')
    parser.add_argument(
        'output_file', type=Path,
        help='Where to save a visualization of bistring migration')
    parser.add_argument(
        '--migration_rate', '-mr', type=float, default=0.5,
        help='Migration rate to visualize (default is 0.5)')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
