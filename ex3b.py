import numpy as np
import scipy as sp
from scipy import linalg
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt

# kB = 3.167e-6  # Eh*K-1
# hbar = 1.
# e = 1.
# T = 0
# ev = 0.0367493
# V = 0.3 * ev*e
# amper = 6.623618e-3

kB = 3.167e-6  #
hbar = 1
e = 1
T = 0
ev = 0.0367493
V = 0.3 / 27.211386

N_L = 300
N_R = 300
N_ML = 50
N_MR = 50
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
    AB = linalg.block_diag(A, B)
    AB[N_A, N_A - 1] = a
    AB[N_A - 1, N_A] = a
    return AB


def fermi_dirac(E, T, mu):
    return 1. / (np.exp((E - mu) / (kB * T)) + 1)
    #return np.heaviside(E-mu, E-mu)

def f_dde(rho, t, h, rho_0_L, rho_0_R, gamma):
    ll = rho[:N_L, :N_L]
    eml = rho[N_L:-N_R, :N_L]
    rl = rho[-N_R:, :N_L]
    rem = rho[-N_R:, N_L:-N_R]
    lem = rho[:N_L, N_L:-N_R]
    lr = rho[:N_L, -N_R:]
    emr = rho[N_L:-N_R, -N_R:]
    rr = rho[-N_R:, -N_R:]
    emem = np.zeros([N_EM, N_EM])
    # rho_0_L_edited = np.copy(ll)
    # np.fill_diagonal(rho_0_L_edited, np.diagonal(rho_0_L))
    # rho_0_R_edited = np.copy(rr)
    # np.fill_diagonal(rho_0_R_edited, np.diagonal(rho_0_R))
    return -1.j*(h @ rho - rho @ h)/hbar - gamma * np.concatenate([np.concatenate([ll-rho_0_L, 0.5*eml, rl], axis=0),
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
    # print(f_dde(rho, t0, h, rho_0_L, rho_0_R, gamma))
    # return rho + dt * f_dde(rho, t0, h, rho_0_L, rho_0_R, gamma)

def I(rho, h):
    h_eml = h[N_L:-N_R, :N_L]
    h_rem = h[-N_R:, N_L:-N_R]
    h_lem = h[:N_L, N_L:-N_R]
    h_emr = h[N_L:-N_R, -N_R:]

    rho_eml = rho[N_L:-N_R, :N_L]
    rho_rem = rho[-N_R:, N_L:-N_R]
    rho_lem = rho[:N_L, N_L:-N_R]
    rho_emr = rho[N_L:-N_R, -N_R:]

    I_L = -(2.j/hbar) * np.trace(h_eml @ rho_lem - rho_eml @ h_lem)
    I_R = -(2.j/hbar) * np.trace(h_emr @ rho_rem - rho_emr @ h_rem)
    return -(e/2.)*(I_L + I_R)


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
# print(h)
L_diag, U_L = sp.linalg.eigh(L)
EM_diag, U_EM = sp.linalg.eigh(EM)
R_diag, U_R = sp.linalg.eigh(R)
U = sp.linalg.block_diag(U_L, U_EM, U_R)
# print(eigen_energies)
L_diag = np.diag(L_diag)
EM_diag = np.diag(EM_diag)
R_diag = np.diag(R_diag)

E_F_L = (np.diag(L_diag)[np.diag(L_diag).shape[0] // 2] + np.diag(L_diag)[np.diag(L_diag).shape[0] // 2 - 1]) / 2
E_F_R = (np.diag(R_diag)[np.diag(R_diag).shape[0] // 2] + np.diag(R_diag)[np.diag(R_diag).shape[0] // 2 - 1]) / 2
mu_L = fermi + 0.5 * V
mu_R = fermi - 0.5 * V
rho_0_EM = np.diag(fermi_dirac(np.diag(EM_diag), T, fermi))
rho_0_L = np.diag(fermi_dirac(np.diag(L_diag), T, mu_L))
rho_0_R = np.diag(fermi_dirac(np.diag(R_diag), T, mu_R))
# rho = np.diag(fermi_dirac(np.diag(h_diag), T, fermi))
rho = sp.linalg.block_diag(rho_0_R, rho_0_EM, rho_0_L)
print(np.diag(rho_0_R))
rho_0_L_wave = np.conjugate(U_L).T @ rho_0_L @ U_L
rho_0_R_wave = np.conjugate(U_R).T @ rho_0_R @ U_R
rho_wave = np.conjugate(U).T @ rho @ U
h_wave = np.conjugate(U).T @ h @ U
# ll = np.conjugate(U_L).T @ h[:N_L, :N_L] @ U_L
# eml = np.conjugate(U_EM).T @ h[N_L:-N_R, :N_L] @ U_L
# rl = np.conjugate(U_R).T @ h[:N_L, -N_R:] @ U_L
# rem = np.conjugate(U_R).T @ h[-N_R:, N_L:-N_R] @ U_EM
# lem = np.conjugate(U_L).T @ h[:N_L, N_L:-N_R] @ U_EM
# lr = np.conjugate(U_L).T @ h[:N_L, -N_R:] @ U_R
# emr = np.conjugate(U_EM).T @ h[N_L:-N_R, -N_R:] @ U_R
# rr = np.conjugate(U_R).T @ h[-N_R:, -N_R:] @ U_R
# emem = np.conjugate(U_EM).T @ h[N_L:-N_R, N_L:-N_R] @ U_EM
#
# h_wave = np.concatenate([np.concatenate([ll, eml, rl], axis=0),
#                            np.concatenate([lem, emem, rem], axis=0),
#                            np.concatenate([lr, emr, rr], axis=0)],
#                            axis=1)
# ll =  np.conjugate(U_L).T @ rho[:N_L, :N_L] @ U_L
# eml = np.conjugate(U_EM).T @ rho[N_L:-N_R, :N_L] @ U_L
# rl = np.conjugate(U_R).T @ rho[:N_L, -N_R:] @ U_L
# rem = np.conjugate(U_R).T @ rho[-N_R:, N_L:-N_R] @ U_EM
# lem = np.conjugate(U_L).T @ rho[:N_L, N_L:-N_R] @ U_EM
# lr = np.conjugate(U_L).T @ rho[:N_L, -N_R:] @ U_R
# emr = np.conjugate(U_EM).T @ rho[N_L:-N_R, -N_R:] @ U_R
# rr = np.conjugate(U_R).T @ rho[-N_R:, -N_R:] @ U_R
# emem = np.conjugate(U_EM).T @ rho[N_L:-N_R, N_L:-N_R] @ U_EM
#
# rho_wave = np.concatenate([np.concatenate([ll, eml, rl], axis=0),
#                            np.concatenate([lem, emem, rem], axis=0),
#                            np.concatenate([lr, emr, rr], axis=0)],
#                            axis=1)
# print(np.diag(rho))
# plt.plot(np.diag(rho))
# plt.show()
current = []
# h_wave = np.conjugate(U).T @ h @ U
timeunit = 41.35649
try:
    for i in range(1500):
        # print(np.trace(rho_wave))
        rho_wave = runge_kutta(rho_wave, timeunit * 1., h_wave, rho_0_L_wave, rho_0_L_wave, timeunit*0)
        # rho = runge_kutta(rho, 1., h, rho_0_L, rho_0_L, 0)
        # rho_wave = np.conjugate(U).T @ rho @ U
        current.append(I(rho_wave, h_wave)*6.623617e-3)
        print(np.trace(rho_wave))
        # current.append(I(rho_wave, h_wave)*1.e15)
        print(current[-1])
except KeyboardInterrupt:
    pass

plt.plot(range(len(current)), current)
plt.show()