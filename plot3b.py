import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def load_file(V, gamma, beta=np.array([-0.2, -0.2, -0.2, -0.2, -0.2]), alpha=np.array([0.0, 0.0, 0.0])):
    file = open(str(V) + ',' + str(gamma) + str(alpha) + str(beta) + ".pkl", 'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

def matrix_format(a, b, N):
    return np.diag(np.ones(N) * a) + np.diag(np.ones(N - 1) * b, 1) + np.diag(np.ones(N - 1) * b, -1)


def construct_transfer(A, B, a):
    N_A = A.shape[0]
    AB = linalg.block_diag(A, B)
    AB[N_A, N_A - 1] = a
    AB[N_A - 1, N_A] = a
    return AB

def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

N=np.array([300, 50, 6, 50, 300])
beta=np.array([-0.2, -0.2, -0.2, -0.2, -0.2])
alpha=np.array([0.0, 0.0, 0.0])
ev = 1. / 27.211386
N_L, N_ML, N_M, N_MR, N_R = N
b_L, b_LM, b_M, b_RM, b_R = beta * ev
a_L, a_M, a_R = alpha * ev
N_EM = N_ML + N_MR + N_M
N_tot = N_EM + N_L + N_R

L = matrix_format(a_L, b_L, N_L)
ML = matrix_format(a_L, b_L, N_ML)
M = matrix_format(a_M, b_M, N_M)
MR = matrix_format(a_R, b_R, N_MR)
R = matrix_format(a_R, b_R, N_R)
EM = construct_transfer(construct_transfer(ML, M, b_LM), MR, b_RM)
h = construct_transfer(construct_transfer(L, EM, b_L), R, b_R)
eigen_energies, _ = eigen(h)
V = 0.01
L_diag, _ = eigen(L)
M_diag, _ = eigen(M)
R_diag, _ = eigen(R)
EM_diag, _ = eigen(EM)
E_F_R = np.median(R_diag)
E_F_L = np.median(L_diag)
mu_L = E_F_L + 0.5 * V
mu_R = E_F_R - 0.5 * V
E_F_R = np.median(L_diag)
E_F_L = np.median(L_diag)
T = 0.
def fermi_dirac(E, T, mu):
    return -(np.heaviside(E-mu, E-mu) - 1.)
rho_0_L = fermi_dirac(L_diag, T, mu_L)
rho_0_R = fermi_dirac(R_diag, T, mu_R)
fermi = np.median(eigen_energies)

rho = np.diag(load_file(0.3, V)[1])
print(rho.sum())
L = rho[:300]
R = rho[-300:]
EM = rho[300:-300]
M = rho[350:356]
# print(rho)
fig = plt.figure()
ax = fig.subplots(1)
ax.set_xlabel(r'$E-E_{f}\left[E_{h}\right]$')
ax.set_ylabel("Occupation")
ax.plot(L_diag, L)
ax.plot(R_diag, R)
ax.plot(R_diag, rho_0_R-fermi)
ax.plot(L_diag, rho_0_L-fermi)

# ax.scatter(M_diag, M)
# ax.plot(EM_diag, EM)
# ax.plot(eigen_energies, rho)


# ax.plot(eigen_energies, rho)
fig.savefig("3b_66.svg")

# fig = plt.figure()
# ax = fig.subplots(1)
# V = 0.3
# for gamma in [0.0, 0.001, 0.01, 0.1, 1.0]:
#     current_i = load_file(V, gamma)[0]
#     ax.plot(current_i, label='$\Gamma='+str(format(gamma, '.3f'))+'\mathtt{fs}^{-1}$')

#
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Current [mA]')
# plt.legend()
# plt.savefig("3b_6.svg")
# plt.clf()


# def matrix_format(a, b, N):
#     return np.diag(np.ones(N) * a) + np.diag(np.ones(N - 1) * b, 1) + np.diag(np.ones(N - 1) * b, -1)
#
#
# def construct_transfer(A, B, a):
#     N_A = A.shape[0]
#     AB = linalg.block_diag(A, B)
#     AB[N_A, N_A - 1] = a
#     AB[N_A - 1, N_A] = a
#     return AB
#
# def eigen(A):
#     eigenValues, eigenVectors = np.linalg.eig(A)
#     idx = np.argsort(eigenValues)[::-1]
#     eigenValues = eigenValues[idx]
#     eigenVectors = eigenVectors[:,idx]
#     return (eigenValues, eigenVectors)
#
# N=np.array([300, 50, 6, 50, 300])
# beta=np.array([-0.2, -0.2, -0.2, -0.2, -0.2])
# alpha=np.array([0.0, 0.0, 0.0])
# ev = 1. / 27.211386
# N_L, N_ML, N_M, N_MR, N_R = N
# b_L, b_LM, b_M, b_RM, b_R = beta * ev
# a_L, a_M, a_R = alpha * ev
# N_EM = N_ML + N_MR + N_M
# N_tot = N_EM + N_L + N_R
#
# L = matrix_format(a_L, b_L, N_L)
# ML = matrix_format(a_L, b_L, N_ML)
# M = matrix_format(a_M, b_M, N_M)
# MR = matrix_format(a_R, b_R, N_MR)
# R = matrix_format(a_R, b_R, N_R)
# EM = construct_transfer(construct_transfer(ML, M, b_LM), MR, b_RM)
# h = construct_transfer(construct_transfer(L, EM, b_L), R, b_R)
# eigen_energies, _ = eigen(h)
# # print(eigen_energies)
# # eigen_energies_x = np.array([np.linspace(eigen_energies[0], eigen_energies[-1], 10000)])0
# vals, bins = np.histogram(eigen_energies, bins=20000, range=[-2.e-2, 2.e-2])
# vals = np.array([vals])
# dbin = (bins[1] - bins[0])/2
# bins = np.array([bins])[:, :-1]+dbin

# for gamma in [0.001, 0.01, 0.1, 1.0]:
#     gamma = gamma / (1.e-15/2.418884e-17)
#     lorenzs = (vals*(1. / (np.pi*gamma*(1+((bins-bins.T)/(gamma))**2.)))).mean(axis=1)
#     plt.plot(bins[0], lorenzs, label='$\Gamma='+str(format(gamma*(1.e-15/2.418884e-17), '.3f'))+'\mathtt{fs}^{-1}$')
# plt.xlabel(r'$E\left[\mathrm{E_{h}}\right]$')
# plt.ylabel(r'$D\left[\mathrm{E_{h}}^{-1}\right]$')
# plt.legend()
# plt.savefig("3b_7.svg")
# plt.show()
# plt.clf()

# for beta in [-0.2, -0.05]:
#     fig = plt.figure()
#     ax = fig.subplots(1)
#     subax = fig.add_axes([0.5, 0.5, 0.38, 0.37])
#     subax.set_xlabel('Bias voltage [V]')
#     subax.set_ylabel('Current [mA]')
#
#     last_I = []
#     for V in [0.1, 0.2, 0.3, 0.4, 0.5]:
#         current_i = load_file(V, 0.01, beta=np.array([-0.2, beta, -0.2, beta, -0.2]))[0]
#         ax.plot(current_i, label=str(format(V, '.1f'))+'V')
#         last_I.append(current_i[-1])
#     subax.scatter([0.1, 0.2, 0.3, 0.4, 0.5], last_I, s=80, facecolors='none', edgecolors='r', label="Driven Liouville")
#     if beta == -0.2:
#         subax.scatter([0.1, 0.2, 0.3, 0.4, 0.5], np.array([0.007803051687284638, 0.015484675274386477, 0.023530742776266947, 0.031146762812429923, 0.03921812977256605]
# ), s=40, facecolors='none', edgecolors='k', label="Sylvester", marker='X')
#     elif beta == -0.05:
#         subax.scatter([0.1, 0.2, 0.3, 0.4, 0.5], np.array(
#             [0.00022322602523425176, 0.0024431526678827804, 0.0031827428672332697, 0.003315521652633859, 0.004110716062617395]
#         ),s=40 , facecolors='none', edgecolors='k', label="Sylvester", marker='X')
#     subax.legend()
#     ax.set_xlabel('Time [fs]')
#     ax.set_ylabel('Current [mA]')
#     ax.legend(loc=4)
#     fig.savefig("3b_8"+str(beta)+".svg")
#     plt.clf()


# fig = plt.figure()
# ax = fig.subplots(1)
# subax = fig.add_axes([0.5, 0.5, 0.38, 0.37])
# subax.set_ylabel('Current [mA]')
# subax.set_xlabel(r'$\alpha_{M}$' + '[eV]')
#
# last_I = []
# a_list = [0.05, 0.0, -0.05]
# for alphas in a_list:
#     current_i = load_file(0.3, 0.01, beta=np.array([-0.2, -0.05, -0.2, -0.05, -0.2]), alpha=np.array([0.0, alphas, 0.0]))[0]
#     ax.plot(current_i, label=str(r'$\alpha_{M}='+str(format(alphas, '.2f')))+'eV$', alpha=0.5)
#     last_I.append(current_i[-1])
# subax.scatter(a_list, last_I, s=80, facecolors='none', edgecolors='r', label="Driven Liouville")
# subax.scatter( [0.05, 0.0, -0.05], np.array(
#             [0.0028989354474228198, 0.0031827428672332697, 0.0028989354474234846]
#         ),s=40 , facecolors='none', edgecolors='k', label="Sylvester", marker='X')
# subax.legend()
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Current [mA]')
# ax.legend(loc=4)
# fig.savefig("3b_9"+".svg")
# plt.clf()