import numpy as np
from scipy import special as sp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def grid(x_min, x_max, h_approx):
    number = int((x_max - x_min) / h_approx)
    grid = np.linspace(x_min, x_max, num=number, endpoint=False)
    h = grid[1] - grid[0]
    return grid, h


def der(f, h, stencils, periodic=False, order=1):
    rolling = []
    for i in range(-stencils // 2 + 1, stencils // 2 + 1):
        rolling.append([np.roll(f, -i)])
    rolling = np.concatenate(rolling, axis=0)
    if stencils == 5:
        if order == 1:
            coefficients = np.transpose([np.array([1. / 12., -2. / 3., 0., 2. / 3., -1. / 12.])])
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
        return np.sum(rolling * coefficients, axis=0) / h ** order
    else:
        middle = (np.sum(rolling * coefficients, axis=0))[stencils // 2:-stencils // 2 + 1]
        forward = []
        backward = []
        for i in range(stencils):
            forward.append([np.roll(f, -i)])
            backward.append([np.roll(f, i)])
        forward = np.concatenate(forward, axis=0)[:, :stencils // 2]
        backward = np.concatenate(backward, axis=0)[:, -stencils // 2 + 1:]
        if stencils == 5:
            if order == 1:
                forward_coe = np.transpose([np.array([-25. / 12., 4., -3., 4. / 3., -1. / 4.])])
            if order == 2:
                forward_coe = np.transpose([np.array([35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12])])
        if stencils == 7:
            if order == 1:
                forward_coe = np.transpose(
                    [np.array([-49. / 20., 6., -15. / 2., 20. / 3., -15. / 4., 6. / 5., -1. / 6.])])
            if order == 2:
                forward_coe = np.transpose(
                    [np.array([-49 / 8, 29, -461 / 8, 62, -307 / 8, 13, -15 / 8])])
        if stencils == 9:
            if order == 1:
                forward_coe = np.transpose(
                    [np.array([-1.0187546202093276e+83 / 3.748374423901926e+82, 8, -14, 56 / 3., -35. / 2., 56. / 5.,
                               -14. / 3., 8. / 7., -1. / 8])])
            if order == 2:
                forward_coe = np.transpose(
                    [np.array(
                        [7.968898772174989e+67, -3.738158576058056e+68, 8.445829173551883e+68, -1.2107350749288884e+69,
                         +1.1747318757695717e+69, -7.670608138130042e+68, +3.238398983443197e+68,
                         -8.004788634475647e+67, +8.815953502150755e+66])]) / 1.3600369039952208e+67
        backward_coe = forward_coe * (-1) ** order
        forward_part = np.sum(forward * forward_coe, axis=0)
        backward_part = np.sum(backward * backward_coe, axis=0)
        r = np.concatenate([forward_part, middle, backward_part], axis=0) / h ** order
        return r


# 2.

def psi(x, p, x0):
    sigma0 = 1.
    return (sigma0 * (2. * np.pi) ** 0.5) ** (-0.5) * np.exp(-((x - x0) ** 2. / (4. * sigma0 ** 2.)) + 1.j * p * x)


def barrier(x, x0, x1):
    return np.heaviside(x - x0, 1) - np.heaviside(x - x1, 1)


# plot1
def test(ff, dd1, dd2, h_list, s):
    d1_list = []
    d2_list = []
    eps = 1.e-6
    for j in h_list:
        x, h = grid(-1, 1, j)
        f = np.imag(ff(x))
        d1 = np.imag(dd1(x))
        d2 = np.imag(dd2(x))
        err_d1 = abs((der(f, h, s, order=1) - d1) / (abs(d1) + eps)) * 100
        err_d2 = abs((der(f, h, s, order=2) - d2) / (abs(d2) + eps)) * 100
        d1_list.append(np.mean(err_d1[s // 2:-s // 2 + 1]))
        d2_list.append(np.mean(err_d2[s // 2:-s // 2 + 1]))
    return d1_list, d2_list


# h_list = np.logspace(-6, -1)
# leg = []
# for i in [5, 7, 9]:
#     leg.append("stencils="+str(i))
# fig, axs = plt.subplots(2)
# def f(x):
#     return np.exp(1.j*x)
# def f1(x):
#     return -np.sin(x)+1.j*np.cos(x)
# def f2(x):
#     return -np.cos(x)-1.j*np.sin(x)
# for i in [5,7,9]:
#     d1i, d2i = test(f, f1, f2, h_list, i)
#     axs[0].loglog(h_list, d1i)
#     axs[1].loglog(h_list, d2i)
# axs[0].set_ylabel("f' mean relative error [%]")
# axs[1].set_ylabel("f'' mean relative error [%]")
# axs[0].legend(leg, loc='upper right')
# # axs[1].legend(leg, loc='upper right')
# fig.suptitle("f(x)=exp(ix): imaginary part")
# plt.xlabel("dx")
# plt.savefig("img.pdf")


def run(x_min, x_max, periodic=False, p=0, mu=0, x_0=25., A=1., B=1., dx=5e-2, dt=0.001, t_max=10, plot=False):
    if not periodic:
        x, h = grid(x_min - 1, x_max + 1, dx)
    else:
        x, h = grid(x_min, x_max, 5e-2)

    def H(f, x, h):
        v = 0.
        if not periodic:
            v += 1000 * (barrier(x, x_max, x_max + 1) + barrier(x, x_min - 1, x_min))
        v_absorb = -1.j * A * (sp.erf(B * (x - x_0)) + 1.) / 2.
        return -0.5 * der(f, h, 9, periodic=True, order=2) + f * (v + v_absorb)

    def f_dde(y, x, t, h):
        return -1.j * H(y, x, h)

    def runge_kutta(psi_0, x, h, dt):
        t0 = 0.
        k1 = dt * f_dde(psi_0, x, t0, h)
        k2 = dt * f_dde(psi_0 + k1 / 2, x, t0 + dt / 2., h)
        k3 = dt * f_dde(psi_0 + k2 / 2, x, t0 + dt / 2., h)
        k4 = dt * f_dde(psi_0 + k3, x, t0 + dt, h)
        return psi_0 + (k1 + 2. * k2 + 2. * k3 + k4) / 6

    y_list = [psi(x, p, mu)]
    y = y_list[0]
    energy_list = [[h * np.conj(y) * H(y, x, h)]]
    normalization_list = [[h * np.conj(y) * y]]


    if plot:
        fig = plt.figure()
        fig.set_tight_layout(True)
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        title = ''
        if periodic:
            title += 'Periodic boundary, '
        else:
            ax.axvline(x_min, c='r', ls='--')
            ax.axvline(x_max, c='r', ls='--')
            title += 'Rigid boundary, '
        title = "p={p}, A={a}, B={b}".format(a=str(A)[:5], b=str(B)[:5], p=str(p)[:5])
        fig.suptitle(title)
        ax.set_ylim([-1, 1])
        ax.set_xlim([x[0], x[-1]])
        ax2.set_ylim([-1.25*10*0.5*p**2., 1.25*10*0.5*p**2.])
        ax2.set_ylabel("Im(V)")
        ax.grid(True, which='both', ls='--')
        ax.axhline(y=0, color='k', alpha=0.75)
        ax.set_axisbelow(True)
        ax2.plot(x, A * (sp.erf(B * (x - x_0)) + 1.) / 2., ls='dashed', color='m')
        ax.set_xlabel("X")
        line1, = ax.plot(x, np.abs(y), label='|Ψ(x)|')
        line2, = ax.plot(x, np.real(y), label='Re(Ψ(x))')
        line3, = ax.plot(x, np.imag(y), label='Imag(Ψ(x))')
        leg = ax.legend()
        leg.remove()
        ax2.add_artist(leg)
        frame_skip = 4  # frame spacings to not plot, for memory reasons

        def animate(i):
            for k in range(frame_skip):
                y = runge_kutta(y_list[0], x, h, dt=dt)
                y_list[0] = y
                energy_list.append([h * np.conj(y) * H(y, x, h)])
                normalization_list.append([h * np.conj(y) * y])
            line1.set_ydata(np.abs(y))
            line2.set_ydata(np.real(y))
            line3.set_ydata(np.imag(y))
            return line1, line2, line3
        ani = FuncAnimation(fig, animate, frames=int((t_max / dt) / frame_skip), blit=True)
        ani.save("p={p}, A={a}, B={b}.gif".format(a=str(A)[:5], b=str(B)[:5], p=str(p)[:5]), dpi=250, writer=PillowWriter(fps=50))
    else:
        for k in range(int(t_max / dt)):
            y = runge_kutta(y_list[0], x, h, dt=dt)
            y_list[0] = y
            energy_list.append([h * np.conj(y) * H(y, x, h)])
            normalization_list.append([h * np.conj(y) * y])
    energies = np.real(np.sum(np.concatenate(energy_list, axis=0), axis=1))
    normalization = np.real(np.sum(np.concatenate(normalization_list, axis=0), axis=1))
    # energy_error = 100. * np.abs((energies[0] - energies) / energies[0])
    # normalization_error = 100. * np.abs((1. - normalization))
    return energies, normalization


def plot_dens(l, m, r, name, steps):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(y=0, color='k', alpha=0.5, )
    ax.axhline(y=1, color='r', alpha=0.75, ls='--')
    ax.semilogy(np.arange(steps+1), l, label='On the left')
    ax.semilogy(np.arange(steps+1), m, label='In the barrier')
    ax.semilogy(np.arange(steps+1), r, label='On the right')
    ax.legend()
    ax.set_ylim((10**-30, 10))
    fig.suptitle("Integrated density")
    plt.xlabel("Time steps [dt]")
    plt.savefig(name+".svg")

p = 5.
# t_max = 5.

# for b in [0.2, 1., 5.]:
#     t_max = 5.
#     if b == 0.2:
#         t_max = 10.
#     energy_error_1, normalization_error_1 = run(-20, 100, periodic=False, p=p, x_0=10, A=0.5 * p ** 2., B=b, dx=5e-2,
#                                                 dt=0.001, t_max=t_max, plot=True)
#
#     steps = len(energy_error_1) - 1
#     fig, axs = plt.subplots(2)
#     axs[0].semilogy(np.arange(steps + 1), energy_error_1, label='Periodic boundary, p=0')
#     axs[0].set_ylabel("Energy")
#     axs[1].semilogy(np.arange(steps + 1), normalization_error_1, label='Periodic boundary, p=0')
#     axs[1].set_ylabel("Normalization")
#
#     # fig.suptitle("Relative errors [%], dt=0.001")
#     plt.xlabel("Time steps [dt]")
#     plt.savefig("a="+str(1)+",b="+str(b)+".svg", bbox_inches='tight')
#
# for a in [0.2, 1., 5.]:
#     t_max = 5.
#     if a == 0.2:
#         t_max = 10.
#     energy_error_1, normalization_error_1 = run(-20, 100, periodic=False, p=p, x_0=10, A=a*0.5*p**2., B=1, dx=5e-2, dt=0.001, t_max=t_max, plot=True)
#
#     steps = len(energy_error_1)-1
#     fig, axs = plt.subplots(2)
#     axs[0].semilogy(np.arange(steps+1), energy_error_1, label='Periodic boundary, p=0')
#     axs[0].set_ylabel("Energy")
#     axs[1].semilogy(np.arange(steps+1), normalization_error_1, label='Periodic boundary, p=0')
#     axs[1].set_ylabel("Normalization")
#
#     # axs[1].legend()
#     # fig.suptitle("Relative errors [%], dt=0.001")
#     plt.xlabel("Time steps [dt]")
#     plt.savefig("a="+str(a)+",b="+str(1.)+".svg", bbox_inches='tight')

energy_error_1, normalization_error_1 = run(-20, 100, periodic=False, p=p, x_0=10, A=0. * p ** 2., B=1, dx=5e-2,
                                            dt=0.001, t_max=5., plot=True)

steps = len(energy_error_1) - 1
fig, axs = plt.subplots(2)
axs[0].plot(np.arange(steps + 1), energy_error_1, label='Periodic boundary, p=0')
axs[0].set_ylabel("Energy")
axs[1].plot(np.arange(steps + 1), normalization_error_1, label='Periodic boundary, p=0')
axs[1].set_ylabel("Normalization")

# fig.suptitle("Relative errors [%], dt=0.001")
plt.xlabel("Time steps [dt]")
plt.savefig("a="+str(0)+".svg", bbox_inches='tight')