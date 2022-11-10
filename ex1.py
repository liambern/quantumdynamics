import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# 1.


def grid(x_min, x_max, h_approx):
    number = int((x_max-x_min) / h_approx)
    grid = np.linspace(x_min, x_max, num=number, endpoint=False)
    h = grid[1] - grid[0]
    return grid, h


def der(f, h, stencils, periodic=False, order=1):
    rolling = []
    for i in range(-stencils//2+1, stencils//2+1):
        rolling.append([np.roll(f, -i)])
    rolling = np.concatenate(rolling, axis=0)
    if stencils == 5:
        if order == 1:
            coefficients = np.transpose([np.array([1./12., -2./3., 0., 2./3., -1./12.])])
        if order == 2:
            coefficients = np.transpose([np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])])
    if stencils == 7:
        if order == 1:
            coefficients = np.transpose([np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])])
        if order == 2:
            coefficients = np.transpose([np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])])
    if stencils == 9:
        if order == 1:
            coefficients = np.transpose(
                [np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])])
        if order == 2:
            coefficients = np.transpose(
                [np.array([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])])
    if periodic:
        return np.sum(rolling * coefficients, axis=0) / h**order
    else:
        middle = (np.sum(rolling * coefficients, axis=0))[stencils//2:-stencils//2+1]
        forward = []
        backward = []
        for i in range(stencils):
            forward.append([np.roll(f, -i)])
            backward.append([np.roll(f, i)])
        forward = np.concatenate(forward, axis=0)[:, :stencils//2]
        backward = np.concatenate(backward, axis=0)[:, -stencils//2+1:]
        if stencils == 5:
            if order == 1:
                forward_coe = np.transpose([np.array([-25. / 12., 4., -3., 4. / 3., -1. / 4.])])
            if order == 2:
                forward_coe = np.transpose([np.array([35/12, -26/3, 19/2, -14/3, 11/12])])
        if stencils == 7:
            if order == 1:
                forward_coe = np.transpose(
                    [np.array([-49. / 20., 6., -15. / 2., 20. / 3., -15. / 4., 6. / 5., -1. / 6.])])
            if order == 2:
                forward_coe = np.transpose(
                    [np.array([-49/8, 29, -461/8, 62, -307/8, 13, -15/8])])
        if stencils == 9:
            if order == 1:
                forward_coe = np.transpose(
                    [np.array([-1.0187546202093276e+83 / 3.748374423901926e+82, 8, -14, 56 / 3., -35. / 2., 56. / 5.,
                               -14. / 3., 8. / 7., -1. / 8])])
            if order == 2:
                forward_coe = np.transpose(
                    [np.array([7.968898772174989e+67, -3.738158576058056e+68, 8.445829173551883e+68,-1.2107350749288884e+69,+1.1747318757695717e+69,-7.670608138130042e+68,+3.238398983443197e+68,-8.004788634475647e+67,+8.815953502150755e+66])]) / 1.3600369039952208e+67
        backward_coe = forward_coe * (-1)**order
        forward_part = np.sum(forward * forward_coe, axis=0)
        backward_part = np.sum(backward * backward_coe, axis=0)
        r = np.concatenate([forward_part, middle, backward_part], axis=0) / h**order
        return r




#2.

def psi(x):
    sigma0 = 1
    x0 = 0.
    p = 10.
    return (sigma0*(2.*np.pi)**0.5)**(-0.5) * np.exp(-((x-x0)**2. / (4. * sigma0**2.)) + 1.j*p*x)
    # return np.exp(-x**2.)


def barrier(x, x0, x1):
    return np.heaviside(x-x0, 1) - np.heaviside(x-x1, 1)


def H(f, x, h):
    # laplace = der(np.real(f), h, 5, periodic=True, order=2) + 1.j * der(np.imag(f), h, 5, periodic=True, order=2)
    v = 1000*(barrier(x, 9, 10) + barrier(x, -10, -9))
    return -0.5 * der(f, h, 9, periodic=True, order=2) + f*v


def f_dde(y, x, t, h):
    return -1.j * H(y, x, h)


def energy(y, x, h):
    return h*np.sum(np.conj(y) * H(y, x, h), axis=1)


def runge_kutta(psi_0, x, h, dt):
    t0 = 0.
    k1 = dt*f_dde(psi_0, x, t0, h)
    k2 = dt*f_dde(psi_0 + k1/2, x, t0 + dt/2., h)
    k3 = dt*f_dde(psi_0 + k2/2, x, t0 + dt/2., h)
    k4 = dt*f_dde(psi_0 + k3, x, t0 + dt, h)
    return psi_0 + (k1+2.*k2+2.*k3+k4)/6


#plot1
def test(ff, dd1, dd2, h_list, s):
    d1_list = []
    d2_list = []
    eps = 0.
    for j in h_list:
        x, h = grid(-1, 1, j)
        f = ff(x)
        d1 = dd1(x)
        d2 = dd2(x)
        err_d1 = abs((der(f, h, s, order=1)-d1)/(d1+eps))*100
        err_d2 = abs((der(f, h, s, order=2)-d2)/(d2+eps))*100
        d1_list.append(np.mean(err_d1[s//2:-s//2+1]))
        d2_list.append(np.mean(err_d2[s//2:-s//2+1]))
    return d1_list, d2_list

h_list = [1.e-1, 5.e-2, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
leg = []
for i in [5, 7, 9]:
    leg.append("stencils="+str(i))
f = np.sin
fig, axs = plt.subplots(2)
def ddddd2(x):
    return -np.sin(x)
for i in [5,7,9]:
    d1i, d2i = test(np.exp, np.exp, np.exp, h_list, i)
    axs[0].loglog(h_list, d1i)
    axs[1].loglog(h_list, d2i)
axs[0].set_ylabel("f' mean relative error [%]")
axs[1].set_ylabel("f'' mean relative error [%]")
axs[0].legend(leg, loc='upper right')
# axs[1].legend(leg, loc='upper right')
fig.suptitle("f(x)=e^x")
plt.xlabel("dx")
plt.show()

# y = psi(x)
# # y = np.sin(x)
# # plt.plot(x, y)
# # plt.plot(x, der(y, h, 5, order=2))
# # plt.show()
# # print(der(y, h, 5, order=1))
# y_list = [[h*np.conj(y)*H(y, x, h)]]
# normalization_list = [[h*np.conj(y)*y]]
# num = len(y)
# p = 0.1
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line1, = ax.plot(x, np.real(y))
# steps = 10000
# dt = 0.001
# for i in range(steps):
#     try:
#         y = runge_kutta(y, x, h, dt=dt)
#         y = y * np.logical_not(np.abs(np.arange(num)-num/2)>(1-p)*num/2).astype(int)
#         line1.set_ydata(np.real(y))
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         y_list.append([h*np.conj(y)*H(y, x, h)])
#         normalization_list.append([h*np.conj(y)*y])
#     except KeyboardInterrupt:
#         pass
#
# energies = np.real(np.sum(np.concatenate(y_list, axis=0), axis=1))
# normalization = np.real(np.sum(np.concatenate(normalization_list, axis=0), axis=1))
#
# energy_error = (energies[0]-energies) / energies[0]
# normalization_error = (1.-normalization) / normalization[0]
# # print(energies)
# # print(normalization)
# # print(energies)
# # plt.plot(np.arange(steps+1)*dt, normalization_error)
# # plt.plot(np.arange(steps+1)*dt, energy_error)
# # plt.show()