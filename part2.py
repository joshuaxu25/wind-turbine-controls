"""MECH 412 project part 2 sample code for students.

James Forbes
2022/03/23
"""
# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
import control


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# %%
# Functions


def circle(x_c, y_c, r):
    """Plot a circle at point (x_c, y_c) of radius r."""
    th = np.linspace(0, 2 * np.pi, 100)
    x = x_c + np.cos(th) * r
    y = y_c + np.sin(th) * r
    return x, y


def robust_nyq(P_nom, W2, wmin, wmax, N_w):
    """Robust Nyquist plot.
    Can be use to plot the nominal plant P(s) with `uncertainty'
    cirlces, or plot L(s) with `uncertainty circles'."""

    # Plot Nyquist plot, output if stable or not
    w_shared = np.logspace(wmin, wmax, N_w)
    # Call control.nyquist to get the count of the -1 point
    count_P_nom = control.nyquist_plot(P_nom, omega=w_shared, plot=False)

    # Set Nyquist plot up
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$Re$')
    ax.set_ylabel(r'$Im$')
    ax.plot(-1, 0, '+', color='C3')

    # Nominal plant magnitude and phase
    mag_P_nom, phase_P_nom, _ = control.bode(P_nom, w_shared, plot=False)
    Re_P_nom = mag_P_nom * np.cos(phase_P_nom)
    Im_P_nom = mag_P_nom * np.sin(phase_P_nom)

    # Plot Nyquist plot
    ax.plot(Re_P_nom, Im_P_nom, '-', color='C3')    
    # ax.plot(Re_P_nom, -Im_P_nom, '-', color='C3')

    number_of_circles = 50  # this can be changed
    w_circle = np.geomspace(10**wmin, 10**wmax, number_of_circles)
    mag_P_nom_W2, _, _ = control.bode(P_nom * W2, w_circle, plot=False)
    mag_P_nom, phase_P_nom, _ = control.bode(P_nom, w_circle, plot=False)
    Re_P_nom = mag_P_nom * np.cos(phase_P_nom)
    Im_P_nom = mag_P_nom * np.sin(phase_P_nom)
    for i in range(w_circle.size):
        x, y = circle(Re_P_nom[i], Im_P_nom[i], mag_P_nom_W2[i])
        ax.plot(x, y, color='C1', linewidth=0.75)

    return count_P_nom, fig, ax


# %%
# Common parameters
# Laplace variable
s = control.tf('s')

# Bode plot frequency bounds and number of points
N_w = 500
w_shared = np.logspace(-3, 3, N_w)


# %%
# Create nominal transfer function
P = 1.206 / (s + 0.2898)  # Nominal plant


# %%
# Optimised W2
W2 = (2319 * s**4 + 63590 * s**3 + 517100 * s**2 + 1294000 * s + 2975000) / (1283 * s**4 + 137500 * s**3 + 5265000 * s**2 + 14860000 * s + 10910000)

mag_abs, _, w = control.bode(W2, w_shared, plot=False)
mag_dB_W2 = 20 * np.log10(mag_abs)

fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).
ax.semilogx(w, mag_dB_W2, '-', color='C0', label=r'$|W_2(j \omega)|$')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('.pdf')





# %%
# W1, basic robust performance requirements
gamma_r = 0.02 # tolerance
tau_a = 4.4 # drop off frequency

W1 = 1/gamma_r * (1 / (s / tau_a + 1) )**3  # dummy value, you change

W1_mag_abs, _, w = control.bode(W1, w_shared, plot=False)
P_mag_abs, _, w = control.bode(P, w_shared, plot=False)


W1_mag_dB = 20 * np.log10(W1_mag_abs)
P_mag_dB = 20 * np.log10(P_mag_abs)

fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).
gamma_r_arr = np.full(w[w <= 4.4].size ,20*np.log10(1/0.02))

ax.semilogx(w, W1_mag_dB, '-', color='C0', label=r'$|W_1(j \omega)|$')
ax.semilogx(w[w <= 4.4], gamma_r_arr, '-', color='C1', label=r'$|1/\gamma_r|$')
ax.semilogx(w, P_mag_dB, '-', color='C2', label=r'$|P(s)|$')



ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('.pdf')

# %%
# Control design via loopshaping
kp = 30

# C1 is a lag controller
# k_lag = 50
b1 = 55
a1 = 4
C1 = kp * (s/b1 + 1) / (s/a1 + 1)

# C2 is a lead Controller
b2 = 0.3
a2 = 3
C2 = (s/b2 + 1) / (s/a2 + 1)

# C3 is a lead Controller
# b3 = 100
# a3 = 1
# C3 = (s/b3 + 1) / (s/a3 + 1)

C = C1 * C2
print(f'C = {C}\n')

W1 = 1/gamma_r * (1 / (s / tau_a + 1) )**3  # dummy value, you change

# W1_mag_abs, _, w = control.bode(W1, w_shared, plot=False)
P_mag_abs, _, w = control.bode(P, w_shared, plot=False)
# L1_mag_abs, _, w = control.bode(C1*P, w_shared, plot=False)
L_mag_abs, _, w = control.bode(C*P, w_shared, plot=False)


# W1_mag_dB = 20 * np.log10(W1_mag_abs)
P_mag_dB = 20 * np.log10(P_mag_abs)
# L1_mag_dB = 20 * np.log10(L1_mag_abs)
L_mag_dB = 20 * np.log10(L_mag_abs)

fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).

# ax.semilogx(w, W1_mag_dB, '-', color='C0', label=r'$|W_1(j \omega)|$')
ax.semilogx(w[w <= 4.4], gamma_r_arr, '-', color='C1', label=r'$|1/\gamma_r|$')
ax.semilogx(w, P_mag_dB, '-', color='C2', label=r'$|P(s)|$')
# ax.semilogx(w, L1_mag_dB, '-', color='C3', label=r'$|C1(s)P(s)|$')
ax.semilogx(w, L_mag_dB, '-', color='C3', label=r'$|L(s)|$')

ax.legend(loc='best')
fig.tight_layout()


# %%
# Open-loop transfer function
L = control.minreal(P * C)

print(f'L = {L}\n')

# %%
# Gang of four
# T
T = control.minreal(control.feedback(P * C))
# S
S = control.minreal(1 - T)
# PS
PS = control.minreal(P * S)
# CS
CS = control.minreal(C * S)

# %%

W2_mag_abs, _, w = control.bode(1/W2, w_shared, plot=False)
W2_mag_dB = 20 * np.log10(W2_mag_abs)

SC_mag_abs, _, w = control.bode(55.6*S*C, w_shared, plot=False)
SC_mag_dB = 20 * np.log10(SC_mag_abs)

fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).

ax.semilogx(w, W1_mag_dB, '.-', color='C0', label=r'$|W_1(j \omega)|$')
ax.semilogx(w, W2_mag_dB, '-', color='C3', label=r'$|1/W_2(j \omega)|$')
ax.semilogx(w, L_mag_dB, '-.', color='C1', label=r'$|L(j \omega)|$')
ax.semilogx(w, SC_mag_dB, '-.', color='C4', label=r'$55.6|S(j \omega)C(j \omega)|$')

ax.legend(loc='best')
fig.suptitle(r'Frequency conditions $|W_1(j \omega)|<|L(j \omega)| and |L(j \omega)| < |W_2(j \omega)|^-1$ and' '\n' r'$1 < 55.6|S(j \omega)C(j \omega)$|')
fig.tight_layout()
# %%
## loop shaping by desired L
# tau1 = 1/4
# tau2 = 1/6
# tau3 = 2/15
# L = 1/(tau1 * s) * 1/(tau2 * s + 1) * 1/(tau3 * s + 1)
# C = L/P

# %%
# Robust performance evaluation

L1_mag_abs, _, w = control.bode(1+L, w_shared, plot=False)
W2L_mag_abs, _, w = control.bode(W2 * L, w_shared, plot=False)

L1_mag_dB = 20 * np.log10(L1_mag_abs)
W2L_mag_dB = 20 * np.log10(W2L_mag_abs)
W1_W2L_mag_dB = 20 * np.log10(W1_mag_abs + W2L_mag_abs)


fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).


ax.semilogx(w, L1_mag_dB, '-', color='C0', label=r'$|1+L(j \omega)|$')
ax.semilogx(w, W1_W2L_mag_dB, '-', color='C1', label=r'$|W_1(j \omega)|+|W_2(j \omega)L(j|\omega)$')
# ax.semilogx(w, W1_mag_dB, '-', color='C2', label=r'$|W_1(j \omega)|$')
# ax.semilogx(w, W2L_mag_dB, '-', color='C3', label=r'$|W_2(j \omega)L(j \omega)|$')

fig.suptitle("Robust Performance Condition Plot")
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('.pdf')

print(f'S = {S}')
print(f'CS = {CS}')
print(f'PS = {PS}')
print(f'T = {T}\n')

# %%
# Nyquist plot of L
count, fig, ax = robust_nyq(L, W2, 1, 3, 250)
fig.suptitle("Robust Nyquist Plot")
fig.tight_layout()
# fig.savefig('.pdf')

# Individual S and T
mag_abs, _, w = control.bode(S, w_shared, plot=False)
mag_dB_S = 20 * np.log10(mag_abs)

mag_abs, _, w = control.bode(T, w_shared, plot=False)
mag_dB_T = 20 * np.log10(mag_abs)

fig, ax = plt.subplots()
fig.suptitle("Sensitivity and Complementary Sensitivity Plot")
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).
ax.semilogx(w, mag_dB_S, '--', color='C3', label=r'$|S(j \omega)|$')
ax.semilogx(w, mag_dB_T, '-.', color='C4', label=r'$|T(j \omega)|$')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('.pdf')

# Bode magnitude plot of P(s), C(s), and L(s)
mag_abs, _, w = control.bode(P, w_shared, plot=False)
mag_dB_P = 20 * np.log10(mag_abs)

mag_abs, _, w = control.bode(C, w_shared, plot=False)
mag_dB_C = 20 * np.log10(mag_abs)

mag_abs, _, w = control.bode(L, w_shared, plot=False)
mag_dB_L = 20 * np.log10(mag_abs)

fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).
ax.semilogx(w, mag_dB_P, '-', color='C0', label=r'$|P(j \omega)|$')
ax.semilogx(w, mag_dB_C, '--', color='C1', label=r'$|C(j \omega)|$')
ax.semilogx(w, mag_dB_L, '-.', color='C2', label=r'$|L(j \omega)|$')
fig.suptitle("P(s), C(s), and L(s) plots")

ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('.pdf')

# %%
# figure for W1(jw)S(jw) and W2(jw)T(jw)
mag_abs, _, w = control.bode(S*W1, w_shared, plot=False)
mag_dB_SW1 = 20 * np.log10(mag_abs)

mag_abs, _, w = control.bode(T*W2, w_shared, plot=False)
mag_dB_TW2 = 20 * np.log10(mag_abs)

ref1 = np.full(w.size,1)

fig, ax = plt.subplots()
# fig.set_size_inches(8.5, 11, forward=True)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
# Magnitude plot (dB).
ax.semilogx(w, ref1, '--', color='C3', label=r'$1$')
ax.semilogx(w, mag_dB_SW1, '--', color='C2', label=r'$|W1(j \omega)S(j \omega)|$')
ax.semilogx(w, mag_dB_TW2, '-.', color='C4', label=r'$|W2(j \omega)T(j \omega)|$')
ax.legend(loc='best')
fig.suptitle("Nominal performance and Internal BIBO stability conditions plots")
fig.tight_layout()

# %%
# Simulate closed-loop system.
# Load profile to track
data_ref = np.loadtxt('ref.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,
                        usecols=(0, 1, ))

# Extract time and reference data
t = data_ref[:, 0]
r = data_ref[:, 1]

# For the purposes of testing the controller C(s), P_true(s) is considered the
# ``true" plant.
A_plant = np.array([[-4.64500000e+03, 1.06793820e+03, -5.56992280e+02,
         8.44638500e+01, 9.91716000e+01, -2.64000000e-01],
        [-1.00000000e+04, 4.47041308e-13, 5.59840915e-13,
        -4.87948758e-13, -8.19131992e-14, 1.27258391e-12],
        [ 0.00000000e+00, -1.00000000e+03, -1.47672753e-13,
         1.56997853e-13, -2.43528124e-13, -2.11672377e-13],
        [ 0.00000000e+00, 0.00000000e+00, -1.00000000e+03,
        -1.28128662e-13, 1.75012746e-13, 1.29344491e-13],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         1.00000000e+02, 6.23467520e-14, -5.04317226e-14],
        [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         0.00000000e+00, -1.00000000e+02, -1.60597507e-13]])
B_plant = np.array([[0.01],
                    [0.],
                    [0.],
                    [0.],
                    [0.],
                    [0.]])
C_plant = np.array([[1380.0, 275.36, 1044.8426, -296.20406, -213.4296, 139.568]])
D_plant = np.array([[0]])
P_true = control.ss2tf(A_plant, B_plant, C_plant, D_plant)  # Do NOT change this!

# Compute closed-loop TFs using ``true" plant.
T_true = control.minreal(control.feedback(P_true * C))

# Forced response of each system
_, y = control.forced_response(T_true, t, r)
e = r-y  # dummy value, you change
_, u = control.forced_response(C, t, e) # dummy value, you change

# Percent of full scale
pFS = e/11.12  # dummy value, you change

# Metrics
e_mean = np.mean(e)
e_std = np.std(e)
e_max = np.max(np.abs(e))
pFS_max = np.max(np.abs(pFS))
u_max = np.max(np.abs(u))

print(f"mean error is {e_mean}.")
print(f"standard deviation of error is {e_std}.")
print(f"maximum error is {e_max}.")
print(f"highest percent of full scale is {pFS_max}.")
print(f"maximum controller output is {u_max}.")


# Plot forced response
fig, ax = plt.subplots(3, 1)
fig.set_size_inches(8.5, 11, forward=True)
ax[0].set_ylabel(r'Force (kN)')
ax[1].set_ylabel(r'Voltage (V)')
ax[2].set_ylabel(r'Perecent of Fullscale ($\%$)')
# Plot data
ax[0].plot(t, y, label=r'$y(t)$ (kN)', color='C0')
ax[0].plot(t, r, '--', label=r'$r(t)$ (kN)', color='C1')
ax[1].plot(t, u, label=r'$u(t)$ (V)', color='C2')
ax[2].plot(t, pFS, label=r'$\%FS$', color='C3')
for a in np.ravel(ax):
    a.set_xlabel(r'Time (s)')
    a.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('.pdf')


# %%
# Plot
plt.show()
