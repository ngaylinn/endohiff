"""Chart evolved environment fitness over generation.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def main(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.with_name('env_fitness.png')
    else:
        output_file.parent.mkdir(exist_ok=True, parents=True)

    log_data = pl.read_parquet(input_file)
    plt.figure(figsize=(2.667, 2.667))
    sns.relplot(
        data=log_data, x='Generation', y='Fitness', hue='Trial', kind='line')
    plt.savefig(output_file, dpi=150)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Chart the fitness over generations for an environment.')
    parser.add_argument(
        'input_file', type=Path,
        help='An environment evolution log to visualize (env_log.parquet)')
    parser.add_argument(
        '--output_file', '-o', type=Path, default=None,
        help='Where to save the image (defaults to fitness.png)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
