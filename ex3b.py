import numpy as np
import scipy as sp
from scipy import linalg
import pickle as pkl
from numba import jit, njit

kB = 3.167e-6
hbar = 1
e = 1
T = 0
ev = 1. / 27.211386
volt = 1. / 27.211386
seconds = 1.e-15/2.418884e-17

def matrix_format(a, b, N):
    return np.diag(np.ones(N) * a) + np.diag(np.ones(N - 1) * b, 1) + np.diag(np.ones(N - 1) * b, -1)


def construct_transfer(A, B, a):
    N_A = A.shape[0]
    AB = linalg.block_diag(A, B)
    AB[N_A, N_A - 1] = a
    AB[N_A - 1, N_A] = a
    return AB


def fermi_dirac(E, T, mu):
    return -(np.heaviside(E-mu, E-mu) - 1.)

def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def create_rectangular_block_matrix(matrix_shape, block_position, block_shape, submat=1):
    matrix = np.zeros(matrix_shape, dtype=float)
    row_start, col_start = block_position
    row_end, col_end = row_start + block_shape[0], col_start + block_shape[1]
    matrix[row_start:row_end, col_start:col_end] = submat
    return matrix

@njit
def extract(matrix, block_position, block_shape):
    row_start, col_start = block_position
    row_end, col_end = row_start + block_shape[0], col_start + block_shape[1]
    return matrix[row_start:row_end, col_start:col_end]


def run(V, gamma, N=np.array([300, 50, 6, 50, 300]), beta=np.array([-0.2, -0.2, -0.2, -0.2, -0.2]), alpha=np.array([0.0, 0.0, 0.0])):
    N_L, N_ML, N_M, N_MR, N_R = N
    b_L, b_LM, b_M, b_RM, b_R = beta * ev
    a_L, a_M, a_R = alpha * ev
    N_EM = N_ML + N_MR + N_M
    N_tot = N_EM + N_L + N_R

    V = V*volt
    gamma = gamma/seconds
    emem = np.zeros((N_EM, N_EM))
    @jit
    def f_dde(rho_wave, t):
        ll = extract(rho_wave, (0, 0), (N_L, N_L))
        eml = extract(rho_wave, (N_L, 0), (N_EM, N_L))
        rl = extract(rho_wave, (N_L + N_EM, 0), (N_R, N_L))
        rem = extract(rho_wave, (N_L + N_EM, N_L), (N_R, N_EM))
        lem = extract(rho_wave, (0, N_L), (N_L, N_EM))
        lr = extract(rho_wave, (0, N_L + N_EM), (N_L, N_R))
        emr = extract(rho_wave, (N_L, N_L + N_EM), (N_EM, N_R))
        rr = extract(rho_wave, (N_L + N_EM, N_L + N_EM), (N_R, N_R))
        return -1.j * (np.matmul(h_wave, rho_wave) - np.matmul(rho_wave, h_wave)) / hbar - gamma * np.block([
                                                                                        [ll - rho_0_L, 0.5 * lem, lr],
                                                                                        [0.5*eml, emem, 0.5*emr],
                                                                                        [rl, 0.5*rem, rr-rho_0_R]])


    @jit
    def runge_kutta(rho, dt):
        t0 = 0.
        k1 = dt * f_dde(rho, t0)
        k2 = dt * f_dde(rho + k1 / 2, t0 + dt / 2)
        k3 = dt * f_dde(rho + k2 / 2, t0 + dt / 2.)
        k4 = dt * f_dde(rho + k3, t0 + dt)
        return rho + (k1 + 2. * k2 + 2. * k3 + k4) / 6

    shape_N = (N_tot, N_tot)
    # tl = create_rectangular_block_matrix(shape_N, (0, 0), (N_L, N_L))
    # tm = create_rectangular_block_matrix(shape_N, (0, N_L), (N_L, N_EM))
    # tr = create_rectangular_block_matrix(shape_N, (0, N_L+N_EM), (N_L, N_R))
    # ml = create_rectangular_block_matrix(shape_N, (N_L, 0), (N_EM,N_L))
    # mm = create_rectangular_block_matrix(shape_N, (N_L, N_L), (N_EM,N_EM))
    # mr = create_rectangular_block_matrix(shape_N, (N_L, N_L+N_EM), (N_EM,N_R))
    # bl = create_rectangular_block_matrix(shape_N, (N_L+N_EM, 0), (N_R,N_L))
    # bm = create_rectangular_block_matrix(shape_N, (N_L+N_EM, N_L), (N_R,N_EM))
    # br = create_rectangular_block_matrix(shape_N, (N_L+N_EM,  N_L+N_EM), (N_R,N_R))


    L = matrix_format(a_L, b_L, N_L)
    ML = matrix_format(a_L, b_L, N_ML)
    M = matrix_format(a_M, b_M, N_M)
    MR = matrix_format(a_R, b_R, N_MR)
    R = matrix_format(a_R, b_R, N_R)
    EM = construct_transfer(construct_transfer(ML, M, b_LM), MR, b_RM)
    h = construct_transfer(construct_transfer(L, EM, b_L), R, b_R)
    eigen_energies, _ = eigen(h)
    fermi = np.median(eigen_energies)

    L_diag, U_L = eigen(L)
    ML_diag, _ = eigen(ML)
    EM_diag, U_EM = eigen(EM)
    M_diag, _ = eigen(M)
    MR_diag, _ = eigen(MR)

    R_diag, U_R = eigen(R)
    U_L = np.asmatrix(U_L)
    U_EM = np.asmatrix(U_EM)
    U_R = np.asmatrix(U_R)

    U = np.asmatrix(sp.linalg.block_diag(U_L, U_EM, U_R))

    E_F_R = np.median(L_diag)
    E_F_L = np.median(L_diag)
    rho_0_EM = np.diag(fermi_dirac(EM_diag, T, fermi))
    rho_0_M = np.diag(fermi_dirac(M_diag, T, fermi))
    h_wave = U.H @ h @ U

    h_LM = extract(h_wave, (0, N_L), (N_L, N_EM))
    h_ML = extract(h_wave, (N_L, 0), (N_EM, N_L))
    h_MR = extract(h_wave, (N_L, N_EM+N_L), (N_EM, N_R))
    h_RM = extract(h_wave, (N_EM+N_L, N_L), (N_R, N_EM))

    @jit
    def I(rho):
        rho_LM = extract(rho, (0, N_L), (N_L, N_EM))
        rho_ML = extract(rho, (N_L, 0), (N_EM, N_L))
        rho_MR = extract(rho, (N_L, N_EM + N_L), (N_EM, N_R))
        rho_RM = extract(rho, (N_EM + N_L, N_L), (N_R, N_EM))
        I_L = -(2.j/hbar) * np.trace(h_ML @ rho_LM - rho_ML @ h_LM)
        I_R = -(2.j/hbar) * np.trace(h_MR @ rho_RM - rho_MR @ h_RM)
        return (e/2.)*(I_L - I_R)*6.623617e-3*1000

    mu_L = E_F_L + 0.5 * V
    mu_R = E_F_R - 0.5 * V
    rho_0_L = np.diag(fermi_dirac(L_diag, T, mu_L))
    rho_0_R = np.diag(fermi_dirac(R_diag, T, mu_R))
    rho = linalg.block_diag(rho_0_L, rho_0_EM, rho_0_R)
    current = []
    try:
        for i in range(3000):
            rho = runge_kutta(rho, seconds)
            current.append(I(rho))
            print(i)
            print(np.trace(rho))
            print(current[-1])
    except KeyboardInterrupt:
        pass

    f = open(str(V/volt) + ',' + str(gamma*seconds) + str(alpha)+ str(beta)+".pkl", "wb")
    pkl.dump([current, rho], f)
    f.close()

for betas in [-0.2, -0.05]:
    for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
        run(V=v, gamma=0.01, beta=np.array([-0.2, betas, -0.2, betas, -0.2]), alpha=np.array([0.0, 0.0, 0.0]))

for alphas in [-0.05, 0.0, 0.05]:
    run(V=0.3, gamma=0.01, beta=np.array([-0.2, -0.05, -0.2, -0.05, -0.2]), alpha=np.array([0.0, alphas, 0.0]))

# def I_syl(gamma):
#     A = -1.j*h_wave/hbar-gamma*(tl+br)/2.
#     B = 1.j*h_wave/hbar-gamma*(tl+br)/2.
#     C = -gamma*create_rectangular_block_matrix(shape_N, (0, 0), (N_L, N_L), rho_0_L_wave)-gamma*create_rectangular_block_matrix(shape_N, (N_L+N_EM,  N_L+N_EM), (N_R, N_R), rho_0_R_wave)
#     return I(linalg.solve_sylvester(A, B, C))



