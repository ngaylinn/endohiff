from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import taichi as ti

from .. import constants

# Override some global constants so we can visualize smaller bit strings, which
# are easier to visualize than the large bit strings we actually evolved. Note
# that we override this value *before* importing other modules that depend on
# it, so they see the updated value.
constants.BITSTR_POWER = 3
constants.BITSTR_LEN = 2 ** constants.BITSTR_POWER
NUM_VALUES = 2**constants.BITSTR_LEN

from ..bitstrings.fitness import score_hiff


@ti.kernel
def compute_hiffs(hiffs: ti.template()):
    for i in range(NUM_VALUES):
        hiffs[i]  = score_hiff(i)


def main(output_file):
    """Visualize the HIFF score for all 8-bit bit strings.
    """
    ti.init(ti.cuda)
    hiffs = ti.field(ti.float16, shape=NUM_VALUES)
    compute_hiffs(hiffs)

    df = pl.DataFrame({
        'Bistrings (00000000 to 11111111)': np.arange(NUM_VALUES),
        'HIFF Score': hiffs.to_numpy()
    })
    sns.relplot(
        data=df, x='Bistrings (00000000 to 11111111)',
        y='HIFF Score', kind='line', aspect=3
    ).set(xticklabels=[]).tick_params(bottom=False)
    plt.savefig(output_file, dpi=600)

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
