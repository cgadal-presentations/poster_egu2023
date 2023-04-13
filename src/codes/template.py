import os
import numpy as np
import cmocean as cmo
import cmasher as cma
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rcParams
from scipy.constants import golden
# \the\textwidth


# %%
# Default parameters (matplotlib.rcParams)
# -------------------------------------
style.use('tableau-colorblind10')

# reset matplotlib to default
# rcParams.update(rcParamsDefault)  # reset

# constants
inches_per_cm = 0.3937
regular_aspect_ratio = 1/golden

########################################
#   Figure size
########################################

text_width = 384.  # en pt
text_width = text_width*0.35136*0.1*inches_per_cm  # inches

regular_figure_width = 0.75*text_width
large_figure_width = .99*text_width

regular_figure_size = np.array([1, regular_aspect_ratio])*regular_figure_width
large_figure_size = np.array([1, regular_aspect_ratio])*large_figure_width

graphical_abstract_figure_size = np.array([6, 5])  # cm
graphical_abstract_figure_size = graphical_abstract_figure_size*inches_per_cm  # inches

rcParams['figure.figsize'] = regular_figure_size

# ########################################
# #   colors
# ########################################
color_list = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200',
              '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']  # colorblind friendly colorlist
plt.rcParams['axes.prop_cycle'] = cycler('color', color_list)

cmap_phi = cmo.cm.haline_r
cmap_slope = cmo.cm.ice
cmap_slope2 = cmo.cm.algae
# cmap_slope = cma.ocean

colors = {
    # #### slope
    'Sand120m_Theta0': cmap_slope(0.9),
    'Slope1': cmap_slope(0.8),
    'Slope3': cmap_slope(0.65),
    'Slope5': cmap_slope(0.45),
    'sand80m_H19': cmap_slope(0.3),
    # 'sand80m_H19': 'tab:cyan',
    'Theta7': cmap_slope2(0.8),
    'Theta10': cmap_slope2(0.5),
    'Theta15': cmap_slope2(0.3),
    # #### settling velocity
    'Saline': color_list[5],
    'Silibeads40_70': color_list[1],
    'silibeads40_70m': color_list[1],
    'Silibeads100_200': color_list[2],
    'Silibeads150_250': color_list[-2],
    # 'silibeads200m_300m': color_list[7]
    'silibeads200m_300m': 'tab:purple'
}

# ########################################
# #   font properties
# ########################################
#
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['font.family'] = 'STIXGeneral'
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{amssymb}"]
# rcParams['font.size'] = 9  # default 12

# ########################################
# #   tick properties
# ########################################
#
# rcParams['xtick.direction'] = 'in'  # ticks pointing insid the figure
# rcParams['ytick.direction'] = 'in'
# rcParams['ytick.right'] = 'True'
# rcParams['xtick.top'] = 'True'

# ########################################
# #   constrained_layout properties
# ########################################
#
# rcParams['figure.constrained_layout.h_pad']: 0.015
# rcParams['figure.constrained_layout.hspace']: 0.15
# rcParams['figure.constrained_layout.wspace']: 0.15
# rcParams['figure.constrained_layout.w_pad']: 0.015
#
# ########################################
# #   other properties
# ########################################
# rcParams['savefig.dpi']: 300
# rcParams['figure.dpi']: 300
