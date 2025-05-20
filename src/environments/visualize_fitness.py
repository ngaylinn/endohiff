from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def main(input_filename, output_filename=None):
    if output_filename is None:
        output_filename = input_filename.with_name('environment_fitness.png')

    log_data = pl.read_parquet(input_filename)
    plt.figure(figsize=(2.667, 2.667))
    sns.relplot(
        data=log_data, x='Generation', y='Fitness', hue='Trial', kind='line')
    plt.savefig(output_filename, dpi=150)

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Chart the fitness over generations for an environment.')
    parser.add_argument(
        'input_filename', type=Path,
        help='An environment evolution log to visualize (outer_log.parquet)')
    parser.add_argument(
        '--output_filename', '-o', type=Path, default=None,
        help='Where to save the image (defaults to fitness.png)')
    args = vars(parser.parse_args())

    sys.exit(main(**args))
