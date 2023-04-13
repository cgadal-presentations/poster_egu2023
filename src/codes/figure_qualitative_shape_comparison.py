import glob
import os
import sys
import ast


import cmocean as cmo
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import PyThemes.quarto as Quarto
from uncertainties import unumpy as unp


def load_parameter_file(file_path):
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if 'xmin' in line:
                xmin = int(line.split(' ')[-1])
            if 'xmax' in line:
                xmax = int(line.split(' ')[-1])
            if 'ymax' in line:
                ymax = int(line.split(' ')[-1])
            if 'xdoor' in line:
                xdoor = int(line.split(' ')[-1])
            if 'sigma1' in line:
                sigma1 = float(line.split(' ')[-1])
            if 'sigma2' in line:
                sigma2 = float(line.split(' ')[-1])
            if 'background_zone' in line:
                background_zone = ast.literal_eval(line.split('=')[-1][1:])
            if 'backlight_break' in line:
                backlight_break = ast.literal_eval(line.split('=')[-1][1:])
    return xmax, xmin, ymax, xdoor, sigma1, sigma2, background_zone, backlight_break


def apply_calibration(points, matrix):
    A = np.hstack([np.ones((points.shape[0], 1)), points,
                   points**2, np.expand_dims(points[:, 0]*points[:, 1], 1)])
    return A @ matrix


def mark_quantity(ax, xy1, xy2, text, padx, pady, **kwargs):
    ax.annotate("", xytext=xy1, xy=xy2,
                arrowprops=dict(arrowstyle="<->", shrinkA=0,
                                shrinkB=0, color='k', mutation_scale=10),
                transform=ax.transData)
    xytext = (np.array(xy1) + np.array(xy2))/2
    ax.text(xytext[0] + padx, xytext[1] + pady, text, **kwargs)


sys.path.append('src')

# Paths to adjust before starting scripts
path_gen = '/media/cyril/LaCie_orang/petit_canal'
round_manip = 'round_spring2022'
results_dir = 'Processing/Results'
ref = 'sand80m_H19/'    # directoy to process
path_total = os.path.join(path_gen, round_manip, results_dir, ref)

calibration_dir = os.path.join(path_total, 'calibration/sand80m_H19')
# path where to find the .par file
par_dir = os.path.join(path_total, '../../scripts/image_processing/par_files')
data_dir = os.path.join(path_total, 'shape')

# ######################## Loading data

# calibration matrix
calib_matrix = np.load(os.path.join(calibration_dir, 'Matrix_pix_to_cm.npy'))
dx_px_av = (calib_matrix[1, 0] + calib_matrix[2, 1])/2

# Loading backligh break position
_, xmin, _, _, _, sigma2, _, backlight_break = load_parameter_file(
    os.path.join(par_dir, 'generic.par'))
pad_backlight = 3*sigma2
bounds_backlight = np.array([[backlight_break[0] + xmin - pad_backlight, 600], [
                            backlight_break[1] + xmin + pad_backlight, 600]])
bounds_backlight = apply_calibration(bounds_backlight, calib_matrix)

# Position processes dictinnary with time indexes
Position_processed = np.load(os.path.join(path_total, 'nose_position', 'Position_processed.npy'),
                             allow_pickle=True).item()
# Loading shapes
SHAPES = np.load(os.path.join(
    data_dir, 'av_shapes/Av_shapes.npy'), allow_pickle=True).item()
SHAPES_props = np.load(os.path.join(
    data_dir, 'av_shapes_log/Shape_logs_props.npy'), allow_pickle=True).item()

# Loading initial parameters
Parameters = np.load(os.path.join(path_total, 'Initial_parameters.npy'),
                     allow_pickle=True).item()

# Listing experiments
list_manips = sorted(glob.glob(os.path.join(path_total, 'raw_contours/run*')))

# parameters
ind = -1  # chosing which contour to study, [0] is completly external
th_len_contour = 200
th_height_contour = 13  # [cm], from the top

# ####
runs = sorted(Position_processed.keys())
Volume_fraction = np.array(
    [Parameters[run]['Volume_fraction'] for run in runs])
runs_sorted, phi_sorted = np.array([[run, phi] for phi, run in
                                    sorted(zip(Volume_fraction, runs))]).T
ind = -1
runs_sorted, phi_sorted = runs_sorted[1:], phi_sorted[1:]
#
cmap = cmo.cm.haline_r
log_phi = np.log10(unp.nominal_values(phi_sorted))
colors = cmap((log_phi - log_phi.min())/(log_phi.max() - log_phi.min()))
colors[:, -1] = 1


fig, ax = plt.subplots(1, 1, figsize=(Quarto.fig_width, 0.9*Quarto.fig_height),
                       constrained_layout=True)

#

exemples = ['run01', 'run03', 'run11', 'run19', 'run15', 'run02']
for (run, color, phi) in zip(runs_sorted, colors, phi_sorted):
    if run in exemples:
        ax.plot(np.abs(SHAPES[run]['xcenters']), unp.nominal_values(SHAPES[run]['shape']),
                color=color, label='${:.2fL}$'.format(100*phi))

ax.axvline(10.5, ls='--', color='k')
ax.text(6, 2.5, r'Head', ha='center', va='center')
ax.text(20, 2.5, r'Tail', ha='center', va='center')
ax.set_xlim(-2.5, 100)
ax.set_ylim(0, 16)
ax.legend(
    title=r'Volume fraction, $\phi~[\%]$', loc='upper right', ncol=3)
ax.set_xlabel('Distance behind nose [cm]')
ax.set_ylabel('Height [cm]')
#
fig_dir = '../'
plt.savefig(os.path.join(fig_dir, '{}.svg'.format(sys.argv[0].split(os.sep)[-1].replace('.py', ''))),
            dpi=400)
