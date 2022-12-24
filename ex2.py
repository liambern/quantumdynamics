import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
import matplotlib.patches as patches
hbar = 1.


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


def barrier(x, x0, x1):
    return np.heaviside(x - x0, 1) - np.heaviside(x - x1, 1)


def initial_phi(x, c0, c1, k=1, m=1):
    normalize = ((k*m)**0.5/(np.pi*hbar))**(1./4.)
    expo = np.exp(-((k*m)**0.5/(2.*hbar))*x**2.)
    f0 = normalize*expo
    f1 = normalize*expo*x*(2.*(k*m)**0.5/hbar)**0.5
    return c0*f0 + c1*f1


def exact_phi(x, t, c0, c1, k=1, m=1):
    normalize = ((k*m)**0.5/(np.pi*hbar))**(1./4.)
    expo = np.exp(-((k*m)**0.5/(2.*hbar))*x**2.)
    f0 = normalize*expo
    f1 = normalize*expo*x*(2.*(k*m)**0.5/hbar)**0.5
    in_exp_t = -1.j*((k / m) ** 0.5 / 2.)*t
    return c0*f0*np.exp(in_exp_t) + c1*f1*np.exp(3*in_exp_t)


def run(x_min, x_max, c0, c1, barriers=[], periodic=False, k=1, m=1, dx=5e-2, dk=0., tc=1., sigma=1, dt=0.001, t_max=10, plot=False):
    time = 0.
    if not periodic:
        x, h = grid(x_min - 1, x_max + 1, dx)
    else:
        x, h = grid(x_min, x_max, 5e-2)


    def H(f, x, h, t):
        v = 0.
        for params in barriers: #list of lists, in the form [v, x_min, x_max]
            v += params[0]*barrier(x, params[1], params[2])
        if not periodic:
            v += 1000 * (barrier(x, x_max, x_max + 1) + barrier(x, x_min - 1, x_min))
        v_harmonic = 0.5 * k * x**2. + 0.5 * dk * x**2. * np.exp(-(t-tc)**2./(2.*sigma**2.))
        return -0.5 * der(f, h, 9, periodic=True, order=2) + f * (v + v_harmonic)


    def P(f, x, h):
        return -1.j * hbar * der(f, h, 9, periodic=True, order=2)

    def f_dde(y, x, t, h):
        return -1.j * H(y, x, h, t)

    def runge_kutta(psi_0, x, h, dt):
        t0 = 0.
        k1 = dt * f_dde(psi_0, x, t0, h)
        k2 = dt * f_dde(psi_0 + k1 / 2, x, t0 + dt / 2., h)
        k3 = dt * f_dde(psi_0 + k2 / 2, x, t0 + dt / 2., h)
        k4 = dt * f_dde(psi_0 + k3, x, t0 + dt, h)
        return psi_0 + (k1 + 2. * k2 + 2. * k3 + k4) / 6

    y_list = [initial_phi(x, c0, c1, k=k, m=m)]
    y = y_list[0]
    energy_list = [[h * np.conj(y) * H(y, x, h, time)]]
    normalization_list = [[h * np.conj(y) * y]]
    x_list = [[h * np.conj(y) * x * y]]
    p_list = [[h * np.conj(y) * P(y, x, h)]]
    before_barrier = []
    in_barrier = []
    after_barrier = []
    if len(barriers) > 0:
        start = barriers[0][1]
        end = barriers[-1][2]
        left = barrier(x, x_min, start)
        mid = barrier(x, start, end)
        right = barrier(x, end, x_max)
        before_barrier.append(normalization_list[-1]*left)
        in_barrier.append(normalization_list[-1]*mid)
        after_barrier.append(normalization_list[-1]*right)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = ''
        if periodic:
            title += 'Periodic boundary, '
        else:
            ax.axvline(x_min, c='r', ls='--')
            ax.axvline(x_max, c='r', ls='--')
            title += 'Rigid boundary, '
        title += 'c0={c0}, c1={c1}, duration={t_run} (all variables in atomic units)'.format(c0=str(c0)[:5], c1=str(c1)[:5], t_run=str(t_max))
        fig.suptitle(title)
        ax.set_ylim([-1, 1])
        ax.set_xlim([x[0], x[-1]])
        ax.grid(True, which='both', ls='--')
        ax.axhline(y=0, color='k', alpha=0.75)
        ax.set_axisbelow(True)
        for params in barriers:
            rect = patches.Rectangle((params[1], 0), params[2]-params[1], params[0], edgecolor='m', ls='dashed', facecolor='none')
            ax.add_patch(rect)
        ax.set_xlabel("X")
        line1, = ax.plot(x, np.abs(y), label='|Ψ(x)|')
        line2, = ax.plot(x, np.real(y), label='Re(Ψ(x))')
        line3, = ax.plot(x, np.imag(y), label='Imag(Ψ(x))')
        ax.legend()

        frame_skip = 4  # frame spacings to not plot, for memory reasons

        def animate(i):
            time = 0.
            for k in range(frame_skip):
                y = runge_kutta(y_list[0], x, h, dt=dt)
                time += dt
                y_list[0] = y
                energy_list.append([h * np.conj(y) * H(y, x, h, time)])
                normalization_list.append([h * np.conj(y) * y])
                x_list.append([h * np.conj(y) * x * y])
                p_list.append([h * np.conj(y) * P(y, x, h)])
                if len(barriers) > 0:
                    before_barrier.append(normalization_list[-1] * left)
                    in_barrier.append(normalization_list[-1] * mid)
                    after_barrier.append(normalization_list[-1] * right)
            line1.set_ydata(np.abs(y))
            line2.set_ydata(np.real(y))
            line3.set_ydata(np.imag(y))
            return line1, line2, line3
        ani = FuncAnimation(fig, animate, frames=int((t_max / dt) / frame_skip), blit=True)
        ani.save("max={max},min={min},c0={c0},c1={c1},periodic={periodic}.gif".format(max=str(x_max), min=str(x_min), c0=str(c0)[:5], c1=str(c1)[:5],
                periodic=str(periodic)), dpi=250, writer=PillowWriter(fps=50))
    else:
        for k in range(int(t_max / dt)):
            y = runge_kutta(y_list[0], x, h, dt=dt)
            time += dt
            y_list[0] = y
            energy_list.append([h * np.conj(y) * H(y, x, h, time)])
            normalization_list.append([h * np.conj(y) * y])
            x_list.append([h * np.conj(y) * x * y])
            p_list.append([h * np.conj(y) * P(y, x, h)])
            if len(barriers) > 0:
                before_barrier.append(normalization_list[-1] * left)
                in_barrier.append(normalization_list[-1] * mid)
                after_barrier.append(normalization_list[-1] * right)
    energies = np.real(np.sum(np.concatenate(energy_list, axis=0), axis=1))
    normalization = np.real(np.sum(np.concatenate(normalization_list, axis=0), axis=1))
    mean_x = np.real(np.sum(np.concatenate(x_list, axis=0), axis=1))
    mean_p = np.real(np.sum(np.concatenate(p_list, axis=0), axis=1))
    energy_error = 100. * np.abs((energies[0] - energies) / energies[0])
    normalization_error = 100. * np.abs((1. - normalization))
    if len(barriers) > 0:
        before_barrier = np.real(np.sum(np.concatenate(before_barrier, axis=0), axis=1))
        in_barrier = np.real(np.sum(np.concatenate(in_barrier, axis=0), axis=1))
        after_barrier = np.real(np.sum(np.concatenate(after_barrier, axis=0), axis=1))
    return energy_error, normalization_error, mean_x, mean_p, before_barrier, in_barrier, after_barrier


cases0 = [0., 0.4, 1/2**0.5, 1/2**0.5, np.exp(-0.1*np.pi*1.j)/2**0.5, (1-0.4**2.)**0.5, 1.]
cases1 = [1., (1-0.4**2.)**0.5, 1/2**0.5, -1/2**0.5, 1/2**0.5, 0.4, 0.] ##fixme make that plot will show imagenary nambers
# for i in range(len(cases0)):
#     energy_error, normalization_error, mean_x, mean_p, _, _, _ = run(-5, 5, cases0[i], cases1[i], barriers=[], periodic=False, k=1, m=1, dx=5e-2, dt=0.001, t_max=1, plot=True)
#     print(mean_x)
#     print(mean_p)
energy_error, normalization_error, mean_x, mean_p, _, _, _ = run(-5, 5, 1., 0., barriers=[],
                                                                 periodic=False, k=1, m=1, dx=5e-2, dt=0.001, t_max=1,
                                                                 plot=True, dk=0.1, tc=0.5)
steps = len(energy_error)-1
fig, axs = plt.subplots(2)
axs[0].semilogy(np.arange(steps+1), energy_error, label='Periodic boundary, p=0')
axs[0].set_ylabel("Energy")
axs[1].semilogy(np.arange(steps+1), normalization_error, label='Periodic boundary, p=0')
axs[1].set_ylabel("Normalization")

axs[1].legend()
fig.suptitle("Relative errors [%], dt=0.001")
plt.xlabel("Time steps [dt]")
plt.show()