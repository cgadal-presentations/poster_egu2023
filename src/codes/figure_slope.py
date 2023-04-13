import os
import sys
import numpy as np
import template as tp
import matplotlib.pyplot as plt
from uncertainties import ufloat
import PyThemes.quarto as Quarto
from uncertainties import unumpy as unp
from lmfit import Model
from matplotlib.legend_handler import HandlerTuple


def Slope_Froude_squared(theta, Fr0, c, a):
    return Fr0**2*(np.cos(np.radians(theta)) + c*a*np.sin(np.radians(theta)))


# Paths to adjust before starting scripts
path_gen = '/media/cyril/LaCie_orang/petit_canal'
DATA = {}
dirs = [
    'round_spring2022/Processing/Results/sand80m_H19/',
    # 'round_spring2022/Processing/Results/silibeads40_70m/',
    # 'round_spring2022/Processing/Results/silibeads120m/',
    'round_spring2022/Processing/Results/silibeads200m_300m/',
    'round_winter2021/Processing/Results/Slope1/',
    'round_winter2021/Processing/Results/Slope3/',
    'round_winter2021/Processing/Results/Slope5/',
    # 'round_winter2021/Processing/Results/Slope7/',
    'round_winter2022/Processing/Results/Silibeads40_70/',
    'round_winter2022/Processing/Results/Silibeads100_200/',
    'round_winter2022/Processing/Results/Silibeads150_250/',
    'round_winter2022/Processing/Results/Saline/',
    'round_winter2022/Processing/Results/Sand120m_Theta0/',
]

DATA2 = {}
dirs_2 = [
    'Manips_LEGI/Processing/Results/Theta7/image_processing/',
    'Manips_LEGI/Processing/Results/Theta10/image_processing/',
    'Manips_LEGI/Processing/Results/Theta15/image_processing/',
]

# #### parameters
ind = -1
g = 9.81  # [m/s2]
rho_f = ufloat(0.998, 0.001)*1e3  # [kg/m3]
rho_p = ufloat(2.65, 0.1)*1e3  # [kg/m3]
mu = ufloat(1, 0.1)*10**-3  # water dynamic viscosity [kg/(m·s)]
L_reservoir = ufloat(9.9, 0.2)*1e-2  # [m]
W_reservoir = ufloat(19.4, 0.2)*1e-2  # [m]

for dir in dirs:
    path_total = os.path.join(path_gen, dir)
    # Loading processed nose positions
    Position_processed = np.load(os.path.join(path_total, 'nose_position/Position_processed.npy'),
                                 allow_pickle=True).item()
    # Loading raw nose positions
    nose_positions = np.load(os.path.join(path_total, 'nose_position/nose_positions.npy'),
                             allow_pickle=True).item()
    # Loading initial parameters
    Parameters = np.load(os.path.join(path_total, 'Initial_parameters.npy'),
                         allow_pickle=True).item()

    # ######## creating variable dictionnary
    fmt = dir.split(os.sep)[-2]
    DATA[fmt] = {}
    DATA[fmt]['Position_processed'] = Position_processed
    DATA[fmt]['nose_positions'] = nose_positions
    DATA[fmt]['Parameters'] = Parameters
    #
    DATA[fmt]['runs'] = sorted(Position_processed.keys())
    DATA[fmt]['slope'] = Parameters[DATA[fmt]['runs'][0]]['slope']
    if fmt == 'Silibeads40_70':
        DATA[fmt]['runs'].remove('run08')  # a lot of bubbles
    DATA[fmt]['Volume_fraction'] = np.array([Parameters[run]['Volume_fraction']
                                             for run in DATA[fmt]['runs']])
    DATA[fmt]['velocity'] = np.array([Position_processed[run][ind]['velocity']
                                      for run in DATA[fmt]['runs']])*1e-2  # [m/s]
    if 'Slope' in fmt:
        DATA[fmt]['velocity'] = DATA[fmt]['velocity'] * \
            1.007  # correction from calibration bias
    Vreservoir = np.array([Parameters[run]['V_reservoir'] for run
                           in DATA[fmt]['runs']]) * 1e-6/W_reservoir  # [m2]
    H0 = (Vreservoir/L_reservoir)  # [m]
    #
    DATA[fmt]['H0'] = H0
    # DATA[fmt]['rho_m'] = rho_f*(1 + DATA[fmt]['Volume_fraction']*(rho_p-rho_f)/rho_f)
    DATA[fmt]['rho_m'] = np.array(
        [Parameters[run]['Current density']*1e3 for run in DATA[fmt]['runs']])
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    DATA[fmt]['FROUDES'] = DATA[fmt]['velocity']/DATA[fmt]['vscale']
    DATA[fmt]['timings'] = np.array([Position_processed[run][ind]['times_fit']
                                     for run in DATA[fmt]['runs']])[:, 1]  # [s]
    # mask_time = DATA[fmt]['timings'] > 0
    # mask_vel = (DATA[fmt]['velocity'] > 0)
    mask_time = DATA[fmt]['timings'] > 10.5*L_reservoir/DATA[fmt]['vscale']
    mask_vel = (DATA[fmt]['velocity'] > 0.3*DATA[fmt]['vscale']
                ) & (DATA[fmt]['velocity'] < 0.5*DATA[fmt]['vscale'])
    DATA[fmt]['mask_ok'] = (mask_vel & mask_time)
    DATA[fmt]['set-up'] = 1
    DATA[fmt]['settling_velocity'] = np.array([Parameters[run]['settling_velocity']
                                               for run in DATA[fmt]['runs']])

for dir in dirs_2:
    path_total = os.path.join(path_gen, dir)
    fmt = dir.split(os.sep)[-3]
    DATA[fmt] = {}
    # Loading data
    nose_position_processed = np.load(os.path.join(path_total,
                                                   'nose_position_processed.npy'),
                                      allow_pickle=True).item()
    #
    DATA[fmt]['runs'] = sorted(nose_position_processed.keys())
    DATA[fmt]['H0'] = np.array([nose_position_processed[run]['H0'] for run
                                in DATA[fmt]['runs']])  # [m]
    DATA[fmt]['velocity'] = np.array([nose_position_processed[run]['velocity_fit']
                                      for run in DATA[fmt]['runs']])*1e-2  # [m/s]
    DATA[fmt]['Volume_fraction'] = np.array([nose_position_processed[run]['phi_0']
                                             for run in DATA[fmt]['runs']])  # [-]
    #
    DATA[fmt]['rho_m'] = rho_f * \
        (1 + DATA[fmt]['Volume_fraction']*(rho_p-rho_f)/rho_f)
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    DATA[fmt]['FROUDES'] = DATA[fmt]['velocity']/DATA[fmt]['vscale']
    #
    DATA[fmt]['slope'] = ufloat(float(fmt.replace('Theta', '')), 0.1)
    DATA[fmt]['set-up'] = 2

model = Model(Slope_Froude_squared)
params = model.make_params()
params['Fr0'].set(value=0.45, min=0, max=1, vary=True)
params['c'].set(value=3, min=0, max=10, vary=True)
params['a'].set(value=1, vary=False)

fig, ax = plt.subplots(1, 1, figsize=(Quarto.fig_width,
                                      Quarto.fig_width/1.62),
                       constrained_layout=True)
#
# datas
data_Birman_2007 = np.loadtxt('data_Birman2007.csv', delimiter=',')
#
fmts = ['Sand120m_Theta0', 'Slope1', 'Slope3', 'Slope5', 'sand80m_H19']
Froudes = np.array([np.mean(DATA[fmt]['FROUDES']) for fmt in fmts])
slopes = np.array([DATA[fmt]['slope'] for fmt in fmts])
data1 = np.vstack([slopes, Froudes]).T
#
Froudes = np.array([np.mean(DATA[fmt]['FROUDES'])
                   for fmt in sorted(DATA.keys()) if DATA[fmt]['set-up'] == 2])
slopes = [DATA[fmt]['slope']
          for fmt in sorted(DATA.keys()) if DATA[fmt]['set-up'] == 2]
data2 = np.vstack([slopes, Froudes]).T

# plot and fit
theta_plot = np.linspace(0, 20, 100)
datas = [data_Birman_2007, data1, data2]
markers = ['d', '.', 's']
colors = ['tab:orange', 'tab:blue', 'tab:green']
colors_fit = ['tab:orange', 'tab:blue', 'tab:green']
labels = ['Birman et al. 2007 (numerical)',
          'this study', 'this study (other set-up)']

handles = []
handles_lines = []
for data, marker, color, color_fit, label in zip(datas, markers, colors, colors_fit, labels):
    mask = data[:, 0] <= 30
    y = data[:, 1]**2
    std = unp.std_devs(y[mask])
    result = model.fit(unp.nominal_values(y[mask]), params,
                       theta=unp.nominal_values(data[:, 0][mask]),
                       #    weights=1/std if not (std == 0).all() else None
                       )
    print(result.fit_report(show_correl=False))
    print(result.ci_report())
    #
    if not (std == 0).all():
        a = ax.errorbar(unp.nominal_values(data[:, 0]), unp.nominal_values(data[:, 1]),
                        xerr=unp.std_devs(data[:, 0]), yerr=unp.std_devs(data[:, 1]),
                        fmt=marker, color=color, label=label)
    else:
        a = ax.scatter(data[:, 0], data[:, 1], marker=marker,
                       color=color, edgecolors='none', label=label)
    #
    b, = ax.plot(theta_plot, np.sqrt(Slope_Froude_squared(theta_plot, *result.params.values())),
                 ls='--', lw=1, color=color_fit, label='theory')
    #
    handles.append(a)
    handles.append(b)


handles, labels = ax.get_legend_handles_labels()
unique_labels = list(np.unique(labels))
unique_labels += [unique_labels.pop(1)]
combined_handles = [tuple(
    [h for h, l in zip(handles, labels) if l == label]) for label in unique_labels]

# leg = ax.legend(handles=combined_handles, labels=unique_labels, handlelength=1.5,
#                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, bbox_to_anchor=(1.05, 1),
#                 loc='upper left', borderaxespad=0., framealpha=0.2)

leg = ax.legend(handles=combined_handles, labels=unique_labels, handlelength=1.5,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
leg.get_frame().set_edgecolor((0, 0, 0, 0.2))
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))

ax.set_ylim(0.2, 0.57)
ax.set_xlim(-0.25, 16)
ax.set_xlabel(r'Bottom slope, $\theta~(^\circ)$')
ax.set_ylabel(r'Current velocity, $\mathcal{F}_{r} = u_{\rm c}/u_{0}$')

fig_dir = '../'

# ax.patch.set_alpha(0)
ax.patch.set_alpha(0.2)
fig.patch.set_alpha(0)

plt.savefig(os.path.join(fig_dir, '{}.svg'.format(sys.argv[0].split(os.sep)[-1].replace('.py', ''))),
            dpi=400, facecolor=fig.get_facecolor(), edgecolor='none')
