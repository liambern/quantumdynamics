import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def load_file(V, gamma, beta=np.array([-0.2, -0.2, -0.2, -0.2, -0.2]), alpha=np.array([0.0, 0.0, 0.0])):
    file = open(str(V) + ',' + str(gamma) + str(alpha) + str(beta) + ".pkl", 'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file
V = 0.3
for gamma in [0.0, 0.001, 0.01, 0.1, 1.0]:
    current_i = load_file(V, gamma)[0]
    plt.plot(current_i, label='$\Gamma='+str(format(gamma, '.3f'))+'\mathtt{fs}^{-1}$')
plt.xlabel('Time [fs]')
plt.ylabel('Current [mA]')
plt.legend()
plt.savefig("3b_6.svg")
plt.clf()


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
print(eigen_energies)
eigen_energies_x = np.array([np.linspace(eigen_energies[0], eigen_energies[-1], 10000)])

for gamma in [0.001, 0.01, 0.1, 1.0]:
    gamma = gamma / (1.e-15/2.418884e-17)
    lorenzs = (1. / (np.pi*gamma*(1+((eigen_energies_x-eigen_energies_x.T)/(gamma+1.e-10))**2.))).mean(axis=0)
    plt.plot(eigen_energies_x[0], lorenzs, label='$\Gamma='+str(format(gamma*(1.e-15/2.418884e-17), '.3f'))+'\mathtt{fs}^{-1}$')
plt.xlabel('E [ha]')
plt.ylabel('rho')
plt.legend()
plt.savefig("3b_7.svg")
plt.show()
plt.clf()