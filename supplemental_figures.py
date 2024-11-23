import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import taichi as ti

import constants

constants.BITSTR_POWER = 3
constants.BITSTR_LEN = 2 ** constants.BITSTR_POWER
constants.NUM_WEIGHTS = constants.BITSTR_LEN - 1
NUM_VALUES = 2**constants.BITSTR_LEN

from hiff import weighted_hiff

ti.init(ti.cuda)

hiffs = ti.field(ti.u32, shape=NUM_VALUES)


@ti.kernel
def compute_hiffs():
    weights = ti.Vector([1.0] * constants.NUM_WEIGHTS)
    for i in range(NUM_VALUES):
        _, hiffs[i]  = weighted_hiff(i, weights)


def save_hiff_diagram():
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
    np.random.seed(42)
    sns.relplot(
        x=0.5*np.random.randn(constants.CARRYING_CAPACITY),
        y=0.5*np.random.randn(constants.CARRYING_CAPACITY),
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
