"""
Name: Danylo Maxymlyuk, Yi Fan Li, Joshua Xu

Uncertainty Bound Code for obtaining the optimal upper bound

"""

# %%
import numpy as np
import control
from matplotlib import pyplot as plt

# %%

# Transfer Functions
s = control.tf('s')

# Nomimal Plant
P_s = (1.206) / (s + 0.2898)

# Off-Nominal Plants that satisfied our selection metric
P_1 = (1.199) / (s + 0.2533)
P_2 = (1.244) / (s + 0.2469)
P_3 = (1.235) / (s + 0.2345)
P_4 = (1.238) / (s + 0.272)

P_5 = (-0.2748*s + 184.5)/(s**2 + 150.1*s + 44.65)
P_6 = (0.3347*s + 121.7)/(s**2 + 99.87*s + 25.21)
P_7 = (0.2391*s + 133.3)/(s**2 + 108.4*s + 29.11)
P_8 = (-0.9376*s + 241.3)/(s**2 + 202.6*s + 56.98)
P_9 = (-0.05349*s + 158.7)/(s**2 + 132.8*s + 42.01)



# %%

# Frequency Range 
w = [0.01,10000]

# Bode Plots of each plant
(mag_1, phase_1, omega_1) = control.bode_plot((P_1/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_2, phase_2, omega_2) = control.bode_plot((P_2/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_3, phase_3, omega_3) = control.bode_plot((P_3/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_4, phase_4, omega_4) = control.bode_plot((P_4/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_5, phase_5, omega_5) = control.bode_plot((P_5/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_6, phase_6, omega_6) = control.bode_plot((P_6/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_7, phase_7, omega_7) = control.bode_plot((P_7/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_8, phase_8, omega_8) = control.bode_plot((P_8/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)
(mag_9, phase_9, omega_9) = control.bode_plot((P_9/P_s)-1, omega_limits=w, plot=True, margins=True, method='best', dB=True, Hz=False)

# Create the upper bound of all the bode plots combined
upper_bound = np.maximum.reduce([mag_1,mag_2,mag_3,mag_4,mag_5,mag_6,mag_7,mag_8,mag_9])

# Show the plot
plt.show()

# Plot all the Bode PLots on one plot and the upper bound
plt.xscale("log")
plt.plot(omega_1, 20*np.log10(mag_1), linestyle='dashed',color='blue')
plt.plot(omega_2, 20*np.log10(mag_2), linestyle='dashed',color='blue')
plt.plot(omega_3, 20*np.log10(mag_3), linestyle='dashed',color='blue')
plt.plot(omega_4, 20*np.log10(mag_4), linestyle='dashed',color='blue')
plt.plot(omega_5, 20*np.log10(mag_5), linestyle='dashed',color='blue')
plt.plot(omega_6, 20*np.log10(mag_6), linestyle='dashed',color='blue')
plt.plot(omega_7, 20*np.log10(mag_7), linestyle='dashed',color='blue')
plt.plot(omega_8, 20*np.log10(mag_8), linestyle='dashed',color='blue')
plt.plot(omega_9, 20*np.log10(mag_9), linestyle='dashed',color='blue')
plt.plot(omega_4, 20*np.log10(upper_bound), label ='upper bound',color='purple')



# Set the x axis label
plt.xlabel('Frequency (rad/sec)')
# Set the y axis label
plt.ylabel('Magnitude (dB)')

plt.legend()
plt.show()

# %%
# Custom libraries
import unc_bound

mag_abs_max = upper_bound
mag_dB_max = 20*np.log10(upper_bound)

N_w = 1000
w_shared = np.logspace(-1, 3, N_w)

# Initial guss for W_2(s)
# W_2(s) must be biproper. Will parameterize it as
# W_2(s) = kappa * ( (s / wb1)**2 + 2 * zb1 / wb1 * s + 1 ) / ( (s / wa1)**2 + 2 * za1 / wa1 * s + 1 ) ...  # noqa
# Numerator
wb1 = 9 # GOOD (Controls the dip in the middle)  # rad/s
zb1 = 0.3 # (Controls how pronounced the dip is)
wb2 = 45 # GOOD  # rad/s
zb2 = 1
# Denominator
wa1 = 14 # rad/s
za1 = 1.5 # (Controls how pronounced the peak is)
wa2 = 150 # GOOD  # rad/s
za2 = 2
# DC gain
kappa = 0.3

# Transfer Function Model (Order 4) for W2
s = control.tf('s')
G1 = ((s / wb1)**2 + 2 * zb1 / wb1 * s + 1) / (
    (s / wa1)**2 + 2 * za1 / wa1 * s + 1)
G2 = ((s / wb2)**2 + 2 * zb2 / wb2 * s + 1) / (
    (s / wa2)**2 + 2 * za2 / wa2 * s + 1)
W20 = kappa * G1 * G2

# Compute magnitude part of W_2(s) in absolute units
mag_abs_W20, _, w = control.bode(W20, w_shared, plot=False)
# Copmute magnitude part of W_2(s) in dB
mag_dB_W20 = 20 * np.log10(mag_abs_W20)

# Plot Bode magntude plot in dB and in absolute units
fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'$\gamma(\omega)$ (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'$\gamma(\omega)$ (absolute)')

# Magnitude plot (dB).
ax[0].semilogx(w, mag_dB_max, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_abs_max, '-', color='C4', label='upper bound')
# Magnitude plot (dB).
ax[0].semilogx(w, mag_dB_W20, '-', color='C1', label='initial bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_abs_W20, '-', color='C1', label='initial bound')
ax[0].legend(loc='lower right')
ax[1].legend(loc='upper left')
# fig.savefig(path.joinpath('2nd_order_R_W2_IC.pdf'))
# fig.savefig(path.joinpath('2nd_order_R.pdf'))

# %%
# Optimize
# Design variable initial conditions
x0 = np.array([wa1, za1, wa2, za2, wb1, zb1, wb2, zb2, kappa])

# Size of design variables
n_x = x0.size

# Lower bound on the design variables x. Can't be zero else forming TFs fails.
lb = 1e-4 * np.ones(n_x, )
# Upper bound on the design variables x.
ub = np.ones(n_x, )  # Upper bound on the design variables x.

# Upper bound on DC gain
ub[-1] = 10

# Add more specific lb and ub constrints.
# Don't let the max natural frequency be above omega_max
omega_max = w_shared[-1]
# Don't let the max zeta be above zeta_max
zeta_max = 1
# Don't let the minimum zeta (damping ratio) be less than zeta_min
zeta_min = 0.2
# Form the upper and lower bound np arrays.
for i in range(0, n_x - 1, 2):
    ub[i] = omega_max * ub[i]
    ub[i + 1] = zeta_max * ub[i + 1]
    lb[i + 1] = lb[i + 1] + zeta_min

# Specify max number of iterations.
max_iter = 8000

# Run optimization
x_opt, f_opt, objhist, xlast = unc_bound.run_optimization(
    x0, lb, ub, max_iter, w_shared, mag_abs_max)

# Compute the optimal W_2(s)
W2 = unc_bound.extract_W2(x_opt)

print("The optimal weighting function W_2(s) is ", W2)

# Compute magnitude part of W_2(s) in absolute units
mag_abs_W2, _, _ = control.bode(W2, w_shared, plot=False)
# Compute magnitude part of W_2(s) in dB
mag_dB_W2 = 20 * np.log10(mag_abs_W2)

# Plot the opimization objective function as a function of iterations
fig, ax = plt.subplots()
ax.set_xlabel(r'iteration, k')
ax.set_ylabel(r'objective function, $f(x)$')
ax.semilogy(objhist, '-', color='C3', label=r'$f(x_k)$')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/obj_vs_iter.pdf')

# %%
# Plot the Bode magnitute plot of the optimal W_2(s) tranfer function
fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'$\gamma(\omega)$ (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'$\gamma(\omega)$ (absolute)')
# Magnitude plot (dB).
ax[0].semilogx(w, mag_dB_max, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_abs_max, '-', color='C4', label='upper bound')
# Magnitude plot (dB).
ax[0].semilogx(w, mag_dB_W2, '-', color='C3', label='optimal bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_abs_W2, '-', color='C3', label='optimal bound')
ax[0].legend(loc='lower right')
ax[1].legend(loc='upper left')
# fig.tight_layout()
# fig.savefig(path.joinpath('2nd_order_W2.pdf'))

# %%
# Plot
plt.show()