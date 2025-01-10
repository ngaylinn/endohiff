"""Generates some additional visualizations for explaining this project.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import taichi as ti

import constants

# Override some global constants so we can visualize smaller bit strings, which
# are easier to visualize than the large bit strings we actually evolved. Note
# that we override this value *before* importing other modules that depend on
# it, so they see the updated value.
constants.BITSTR_POWER = 3
constants.BITSTR_LEN = 2 ** constants.BITSTR_POWER
constants.NUM_WEIGHTS = constants.BITSTR_LEN - 1
NUM_VALUES = 2**constants.BITSTR_LEN

from inner_fitness import hiff

ti.init(ti.cuda)


# A lookup table of HIFF scores for all possible bitstrings.
hiffs = ti.field(ti.u32, shape=NUM_VALUES)

@ti.kernel
def compute_hiffs():
    for i in range(NUM_VALUES):
        hiffs[i]  = hiff(i)


def save_hiff_diagram():
    """Visualize the HIFF score for all 8-bit bit strings.
    """
    compute_hiffs()
    df = pl.DataFrame({
        'Bistrings (00000000 to 11111111)': np.arange(NUM_VALUES),
        'HIFF Score': hiffs.to_numpy()
    })
    sns.relplot(
        data=df, x='Bistrings (00000000 to 11111111)',
        y='HIFF Score', kind='line', aspect=3
    ).set(xticklabels=[]).tick_params(bottom=False)
    plt.savefig('output/hiff.png', dpi=600)


def save_migration_diagram():
    """Visualize the spatial distribution of migrating individuals.
    """
    np.random.seed(42)
    sns.relplot(
        x=constants.MIGRATION_RATE*np.random.randn(constants.CARRYING_CAPACITY),
        y=constants.MIGRATION_RATE*np.random.randn(constants.CARRYING_CAPACITY),
        kind='scatter'
    )
    plt.xticks([-1.5, -0.5, 0.5, 1.5], labels=[])
    plt.xticks([-1, 0, 1], labels=['-1', '0', '+1'], minor=True)
    plt.yticks([-1.5, -0.5, 0.5, 1.5], labels=[])
    plt.yticks([-1, 0, 1], labels=['-1', '0', '+1'], minor=True)
    plt.grid()
    plt.tight_layout()
    plt.savefig('output/migration.png')


if __name__ == '__main__':
    save_hiff_diagram()
    save_migration_diagram()
