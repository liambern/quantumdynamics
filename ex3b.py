import numpy as np
import scipy as sp
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt

kB = 3.167e-6#8.617333e-5  # eV*K-1
hbar = 1.#6.582119e-16  # eV*s
e = 1.#1.60217663e-19  # c
ev = 0.0367493
V = 0.3 * ev # [V]
T = 0.0

N_L = 30
N_R = 30
N_ML = 5
N_MR = 5
N_M = 6
N_EM = N_ML + N_MR + N_M

a_L = 0. * ev
a_M = 0. * ev
a_R = 0. * ev

b_L = -0.2 * ev
b_M = -0.2 * ev
b_R = -0.2 * ev
b_LM = -0.2 * ev
b_RM = -0.2 * ev


def matrix_format(a, b, N):
    return np.diag(np.ones(N) * a) + np.diag(np.ones(N - 1) * b, 1) + np.diag(np.ones(N - 1) * b, -1)


def construct_transfer(A, B, a):
    N_A = A.shape[0]
    AB = sp.linalg.block_diag(A, B)
    AB[N_A, N_A - 1] = a
    AB[N_A - 1, N_A] = a
    return AB


def fermi_dirac(E, T, mu):
    return 1. / (np.exp((E - mu) / (kB * T)) + 1)


def f_dde(rho, t, h, rho_0_L, rho_0_R, gamma):
    ll = rho[:N_L, :N_L]
    eml = rho[N_L:N_L + N_EM, :N_L]
    rl = rho[-N_R:, :N_L]
    rem = rho[-N_R:, N_L:N_L + N_EM]
    lem = rho[:N_L, N_L:N_L + N_EM]
    lr = rho[:N_L, -N_R:]
    emr = rho[N_L:N_L + N_EM, -N_R:]
    rr = rho[-N_R:, -N_R:]
    emem = np.zeros([N_EM, N_EM])
    return -1.j * (h @ rho - rho @ h) - gamma * np.concatenate([np.concatenate([ll-rho_0_L, 0.5*eml, rl], axis=0),
                                                                np.concatenate([0.5*lem, emem, 0.5*rem], axis=0),
                                                                np.concatenate([lr, 0.5*emr, rr-rho_0_R], axis=0)],
                                                               axis=1)


def runge_kutta(rho, dt, h, rho_0_L, rho_0_R, gamma):
    t0 = 0.
    k1 = dt * f_dde(rho, t0, h, rho_0_L, rho_0_R, gamma)
    k2 = dt * f_dde(rho + k1 / 2, t0 + dt / 2., h, rho_0_L, rho_0_R, gamma)
    k3 = dt * f_dde(rho + k2 / 2, t0 + dt / 2., h, rho_0_L, rho_0_R, gamma)
    k4 = dt * f_dde(rho + k3, t0 + dt, h, rho_0_L, rho_0_R, gamma)
    return rho + (k1 + 2. * k2 + 2. * k3 + k4) / 6


L = matrix_format(a_L, b_L, N_L)
ML = matrix_format(a_L, b_L, N_ML)
M = matrix_format(a_M, b_M, N_M)
MR = matrix_format(a_R, b_R, N_MR)
R = matrix_format(a_R, b_R, N_R)
EM = construct_transfer(construct_transfer(ML, M, b_LM), MR, b_RM)
h = construct_transfer(construct_transfer(L, EM, b_L), R, b_R)

eigen_energies, _ = sp.linalg.eigh(h)
fermi = (eigen_energies[eigen_energies.shape[0] // 2] + eigen_energies[eigen_energies.shape[0] // 2 - 1]) / 2
h_diag = np.diag(eigen_energies)

L_diag, U_L = sp.linalg.eigh(L)
EM_diag, U_EM = sp.linalg.eigh(EM)
R_diag, U_R = sp.linalg.eigh(R)
U = sp.linalg.block_diag(U_L, U_EM, U_R)

L_diag = np.diag(L_diag)
EM_diag = np.diag(EM_diag)
R_diag = np.diag(R_diag)

E_F_L = (np.diag(L_diag)[np.diag(L_diag).shape[0] // 2] + np.diag(L_diag)[np.diag(L_diag).shape[0] // 2 - 1]) / 2
E_F_R = (np.diag(R_diag)[np.diag(R_diag).shape[0] // 2] + np.diag(R_diag)[np.diag(R_diag).shape[0] // 2 - 1]) / 2
mu_L = E_F_L + 0.5 * V
mu_R = E_F_R - 0.5 * V

rho_0_L = np.diag(fermi_dirac(np.diag(L_diag), T, mu_L))
rho_0_R = np.diag(fermi_dirac(np.diag(R_diag), T, mu_R))
rho = np.diag(fermi_dirac(np.diag(h_diag), T, fermi))

rho_0_L_wave = np.conjugate(U_L).T @ rho_0_L @ U_L
rho_0_R_wave = np.conjugate(U_R).T @ rho_0_R @ U_R
rho_wave = np.conjugate(U).T @ rho @ U
h_wave = np.conjugate(U).T @ h @ U
print(h_wave)
current = []
print(rho_wave)
for i in range(10000):
    rho_wave = runge_kutta(rho_wave, 1.e-15, h_wave, rho_0_L_wave, rho_0_L_wave, 1e-15)
    # current.append((2*b_L*e/hbar)*np.trace(np.imag(U @ rho_wave @ np.conjugate(U).T)))
    current.append(np.trace(f_dde(rho_wave, 0, h_wave, rho_0_L_wave, rho_0_L_wave, 1e-15)))
plt.plot(range(10000), current)
plt.show()