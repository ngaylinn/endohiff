import matplotlib.pyplot as plt
import seaborn as sns

from .constants import ENV_NAMES

VIABILITY_PALETTE = 'mako'
BITSTRING_PALETTE = 'rocket'
ENV_NAME_PALETTE = 'colorblind'
FITNESS_DELTA_PALETTE = 'Spectral'

ENV_NAME_COLORS = {
    env_name: sns.color_palette(ENV_NAME_PALETTE)[i]
    for i, env_name in enumerate(ENV_NAMES)
}


def frameless_figure():
    fig = plt.figure(frameon=False, figsize=(16, 9))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig
