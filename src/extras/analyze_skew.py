"""Verify there's no preference for 0s or 1s when evolving to solve hiff.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import skew
import seaborn as sns
import taichi as ti
from tqdm import trange

from src.constants import BITSTR_LEN, INNER_GENERATIONS
from src.environments.util import STATIC_ENVIRONMENTS, make_env_field
from src.bitstrings.population import BitstrPopulation, make_params_field


# Set these as high as you like to improve statistical significance.
# NUM_PARALLEL determines how many trials to run at once. Set this as high as
# your GPU can handle.
TOTAL_TRIALS = 10_000
NUM_PARALLEL = 25


def count_ones(bitstrs):
    result = np.zeros_like(bitstrs)
    for b in range(BITSTR_LEN):
        mask = 1 << b
        result += (bitstrs & mask) // mask
    return result


def main(output_file):
    ti.init(ti.cuda)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    env = make_env_field(NUM_PARALLEL)
    env.from_numpy(STATIC_ENVIRONMENTS['baym'](NUM_PARALLEL))
    params = make_params_field(NUM_PARALLEL)
    bitstr_population = BitstrPopulation(NUM_PARALLEL)

    # Keep running simulations until we've run TOTAL_TRIALS, and count up the
    # number of ones found in each evolved bitstring.
    partial_counts = []
    for t in trange(TOTAL_TRIALS // NUM_PARALLEL):
        # Evolve some bitstrings, then grab the ones that are still alive in
        # the final generation and count up how many 1 bits they have.
        bitstr_population.evolve(env, params)
        bitstrs = bitstr_population.get_logs(0).filter(
            (pl.col('Generation') == INNER_GENERATIONS - 1) &
            (pl.col('alive') == True)
        )['bitstr'].to_numpy()
        partial_counts.append(count_ones(bitstrs))
    one_counts = np.concatenate(partial_counts)

    # Render a histogram for visual confirmation
    sns.displot(x=one_counts, kind='hist', discrete=True)
    plt.savefig(output_file)

    # Print some stats about the distribution.
    print(f'Median == {np.median(one_counts):0.5f} (32 is expected)')
    print(f'Skew == {skew(one_counts):0.5f} (0 is symmetrical)')

    # Indicate the program completed successfully.
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Check for 0 / 1 bias when evolving bitstrings.')
    parser.add_argument(
        'output_file', type=Path,
        help='Where to save a visualization of 0 / 1 skew')
    args = vars(parser.parse_args())
    sys.exit(main(**args))
