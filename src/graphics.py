import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import ENV_NAMES

VIABILITY_PALETTE = 'mako'
BITSTR_PALETTE = 'rocket'
FITNESS_DELTA_PALETTE = 'Spectral'

# Make sure that each named environment is charted using the same color every
# time.
ENV_NAME_PALETTE = 'colorblind'
ENV_NAME_COLORS = {
    env_name: sns.color_palette(ENV_NAME_PALETTE)[i]
    for i, env_name in enumerate(ENV_NAMES)
}


def frameless_figure():
    """For rendering environmental maps with no decorations.
    """
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig
