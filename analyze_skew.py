"""Verify there's no preference for 0s or 1s when evolving to solve hiff.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import skew
import seaborn as sns
import taichi as ti
from tqdm import trange

from constants import INNER_GENERATIONS, OUTPUT_PATH
from environments import STATIC_ENVIRONMENTS
from inner_population import InnerPopulation, get_default_params


# Set these as high as you like to improve statistical significance.
# NUM_PARALELL determines how many trials to run at once on the GPU. Set this
# as high as your GPU can handle.
TOTAL_TRIALS = 10_000
NUM_PARALLEL = 25


# Set up to run NUM_PARALLEL simulations on the GPU.
ti.init(ti.cuda)
env = STATIC_ENVIRONMENTS['baym'](NUM_PARALLEL)
params = get_default_params(NUM_PARALLEL)
inner_population = InnerPopulation(NUM_PARALLEL)

# Keep running simulations until we've run TOTAL_TRIALS, and count up the
# number of ones found in each evolved bitstring.
partial_counts = []
for t in trange(TOTAL_TRIALS // NUM_PARALLEL):
    inner_population.evolve(env, params)
    partial_counts.append(
        inner_population.get_logs(0).filter(
            (pl.col('Generation') == INNER_GENERATIONS - 1) &
            (pl.col('alive') == True)
        )['one_count'].to_numpy())
one_counts = np.concatenate(partial_counts)

# Render a histogram for visual confirmation
sns.displot(x=one_counts, kind='hist', discrete=True)
plt.savefig(OUTPUT_PATH / 'skew.png')

# Print some stats about the distribution.
print(f'Median == {np.median(one_counts):0.5f} (32 is expected)')
print(f'Skew == {skew(one_counts):0.5f} (0 is symmetrical)')
