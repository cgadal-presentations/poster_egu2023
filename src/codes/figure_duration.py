import os
import sys
import numpy as np
import template as tp
import matplotlib.pyplot as plt
from uncertainties import ufloat
import PyThemes.quarto as Quarto
from uncertainties import unumpy as unp


def Reynolds(U, h, rho, mu):
    return rho*U*h/mu


# Paths to adjust before starting scripts
path_gen = '/media/cyril/LaCie_orang/petit_canal'
DATA = {}
dirs = [
    'round_spring2022/Processing/Results/sand80m_H19/',
    'round_spring2022/Processing/Results/silibeads40_70m/',
    # 'round_spring2022/Processing/Results/silibeads120m/',
    'round_spring2022/Processing/Results/silibeads200m_300m/',
    # 'round_winter2021/Processing/Results/Slope1/',
    # 'round_winter2021/Processing/Results/Slope3/',
    # 'round_winter2021/Processing/Results/Slope5/',
    'round_winter2022/Processing/Results/Silibeads40_70/',
    'round_winter2022/Processing/Results/Silibeads100_200/',
    'round_winter2022/Processing/Results/Silibeads150_250/',
    'round_winter2022/Processing/Results/Saline/',
]

# #### parameters
ind = -1
g = 9.81  # [m/s2]
rho_f = ufloat(0.998, 0.001)*1e3  # [kg/m3]
rho_p = ufloat(2.65, 0.02)*1e3  # [kg/m3]
mu = ufloat(8.90, 1)*10**-4  # water dynamic viscosity [kg/(m·s)]
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
    #
    # ######## creating variable dictionnary
    fmt = dir.split(os.sep)[-2]
    DATA[fmt] = {}
    DATA[fmt]['Position_processed'] = Position_processed
    DATA[fmt]['Parameters'] = Parameters
    #
    DATA[fmt]['runs'] = sorted(Position_processed.keys())
    if fmt == 'Silibeads40_70':
        DATA[fmt]['runs'].remove('run08')  # a lot of bubbles
    DATA[fmt]['Volume_fraction'] = np.array([Parameters[run]['Volume_fraction']
                                             for run in DATA[fmt]['runs']])
    # DATA[fmt]['Stokes_velocity'] = np.array([Parameters[run]['stokes_velocity']
    #                                          for run in DATA[fmt]['runs']])
    # DATA[fmt]['Gen_settling_vel'] = np.array([Parameters[run]['Gen_Settling_velocity']
    #                                          for run in DATA[fmt]['runs']])
    DATA[fmt]['settling_velocity'] = np.array([Parameters[run]['settling_velocity']
                                               for run in DATA[fmt]['runs']])
    #
    Vreservoir = np.array([Parameters[run]['V_reservoir'] for run
                           in DATA[fmt]['runs']]) * 1e-6/W_reservoir  # [m2]
    H0 = (Vreservoir/L_reservoir)  # [m]
    #
    DATA[fmt]['velocity'] = np.array(
        [Position_processed[run][ind]['velocity'] for run in DATA[fmt]['runs']]) * 1e-2  # [m/s]
    DATA[fmt]['H0'] = H0
    DATA[fmt]['rho_m'] = np.array(
        [Parameters[run]['Current density']*1e3 for run in DATA[fmt]['runs']])
    DATA[fmt]['gprime'] = (DATA[fmt]['rho_m'] - rho_f) * g / rho_f
    DATA[fmt]['vscale'] = unp.sqrt(DATA[fmt]['gprime']*DATA[fmt]['H0'])
    DATA[fmt]['REYNOLDS'] = Reynolds(
        DATA[fmt]['vscale'], H0, DATA[fmt]['rho_m'], mu)
    #
    DATA[fmt]['position'] = [Position_processed[run][ind]['position']
                             for run in DATA[fmt]['runs']]
    DATA[fmt]['time'] = [Position_processed[run][ind]['time']
                         for run in DATA[fmt]['runs']]
    DATA[fmt]['t0'] = [Position_processed[run][ind]['virtual_time_origin']
                       for run in DATA[fmt]['runs']]
    DATA[fmt]['x0'] = [Position_processed[run][ind]['virtual_x_origin']
                       for run in DATA[fmt]['runs']]
    DATA[fmt]['time_short'] = [Position_processed[run][ind]['time_short']
                               for run in DATA[fmt]['runs']]
    DATA[fmt]['velocity_ts'] = [Position_processed[run][ind]['velocity_ts']
                                for run in DATA[fmt]['runs']]
    #
    DATA[fmt]['timings'] = np.array([Position_processed[run][ind]['times_fit']
                                     for run in DATA[fmt]['runs']])[:, 1]  # [s]
    # adding +/-1sec uncertainty to end time meas.
    DATA[fmt]['timings'] = unp.uarray(
        DATA[fmt]['timings'], 0.15*DATA[fmt]['timings'])
    #
    mask_time = DATA[fmt]['timings'] > 10.5*L_reservoir/DATA[fmt]['vscale']
    mask_vel = (DATA[fmt]['velocity'] > 0.3*DATA[fmt]['vscale']
                ) & (DATA[fmt]['velocity'] < 0.5*DATA[fmt]['vscale'])
    DATA[fmt]['mask_ok'] = (mask_vel & mask_time)
    #
    DATA[fmt]['i_start_end'] = [Position_processed[run][ind]['indexes_fit']
                                for run in DATA[fmt]['runs']]
    DATA[fmt]['velocity_fit'] = [Position_processed[run][ind]['velocity_fit']
                                 for run in DATA[fmt]['runs']]
    DATA[fmt]['virtual_x_origin'] = [Position_processed[run]
                                     [ind]['virtual_x_origin'] for run in DATA[fmt]['runs']]


fig, ax = plt.subplots(1, 1, figsize=(Quarto.fig_width,
                                      Quarto.fig_width/1.62),
                       constrained_layout=True)
#
fmts = ['sand80m_H19', 'Silibeads40_70',
        'silibeads200m_300m', 'Silibeads100_200', 'Silibeads150_250']
Vss = [DATA[fmt]['settling_velocity'][0] for fmt in fmts]
fmts_sorted, Vss_sorted = np.array([[fmt, vs] for vs, fmt in
                                    sorted(zip(Vss, fmts))]).T


#
for fmt, vs in zip(fmts_sorted, Vss_sorted):
    tscale = L_reservoir/DATA[fmt]['vscale']
    tsed = L_reservoir/(DATA[fmt]['settling_velocity']*1e-2)
    tend = DATA[fmt]['timings']
    #
    x = unp.nominal_values(tscale/tsed)
    xerr = unp.std_devs(tscale/tsed)/2
    y = unp.nominal_values(tend/tsed)
    yerr = unp.std_devs(tend/tsed)/2
    #
    mask = DATA[fmt]['mask_ok']
    #
    a, _, _ = ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask],
                          fmt='.',
                          label=r'${:.2L}$'.format(
                              vs) if not np.isnan(vs.n) else 'Saline',
                          color=tp.colors[fmt])
    ax.set_xscale('log')
    ax.set_yscale('log')


leg1 = ax.legend(title=r'$v_{\rm s}$ [cm/s]',
                 bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., framealpha=0.2)

xth = np.logspace(-3, -1, 100)
a, = ax.plot(xth, 30*xth, color='k', ls='-', lw=1, zorder=-10,
             label=r"$t_{\rm end} = 30t_{0}$"
             )
b = ax.axvline(1/15, color='k', ls=':', lw=1, zorder=-10,
               label=r'$\mathcal{S} = 0.07$'
               )
handles = [a, b]
#
ax.text(0.089, 0.3, 'no \n constant velocity \n regime', rotation=65,
        ha='center', va='center')

c = ax.axhline(0.9, color='k', ls='--', lw=1, zorder=-10,
               label=r'$t_{\rm end} \propto t_{\rm s}$'
               )
handles = [a, b, c]

ax.axvspan(ax.get_xlim()[0], 0.013, zorder=-10,
           color='#fad7a0', linewidth=0, alpha=0.2)
ax.text(0.0065, 0.6, 'negligible settling \n (saline currents limit)',
        ha='center', va='center')
#
ax.axvspan(0.013, 0.0374, zorder=-10,
           color='#d7bde2', linewidth=0, alpha=0.2)
ax.text(0.024, 0.2, 'transition', ha='center', va='center')
#
ax.axvspan(0.0374, 0.0667, zorder=-10,
           color='#16a085', linewidth=0, alpha=0.1)
ax.text(0.05, 0.23, 'settling \n dominated', ha='center', va='center',
        rotation=65)


ax.set_xlim((0.0033, 0.12))
ax.set_ylim((0.1, 1.65))
ax.set_xlabel(
    r"Settling number, $\mathcal{S} = t_{0}/t_{\rm s} = v_{\rm s}/u_{0}$")
ax.set_ylabel(r"Regime duration, $t_{\rm end}/t_{\rm s}$")

leg2 = ax.legend(handles=handles, bbox_to_anchor=(1.05, 0),
                 loc='lower left', borderaxespad=0., framealpha=0.2)
ax.add_artist(leg1)
leg1.get_frame().set_edgecolor('k')
leg2.get_frame().set_edgecolor('k')

fig_dir = '../'
ax.patch.set_alpha(0.2)
fig.patch.set_alpha(0)

plt.savefig(os.path.join(fig_dir, '{}.svg'.format(sys.argv[0].split(os.sep)[-1].replace('.py', ''))),
            dpi=400, facecolor=fig.get_facecolor(), edgecolor='none')
