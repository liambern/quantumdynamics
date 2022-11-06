import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# 1.
x_min = -1.
x_max = 1.
h_approx = 1.e-4
number = int((x_max-x_min) / h_approx)
h = (x_max-x_min) / (number - 1)
grid = np.linspace(x_min, x_max, num=number)
print(grid[1] - grid[0] - h)

def f(x):
    return np.exp(x)

def d1(f, stencils):
    h = f[1] - f[0]
    forward = []
    backward = []
    for i in range(stencils):
        forward.append([np.roll(f, -i)])
        backward.append([np.roll(f, i)])
    forward = np.concatenate(forward, axis=0)[:, :-stencils]
    backward = np.concatenate(backward, axis=0)[:, -stencils:]
    if stencils == 5:
        forward_coe = np.transpose([np.array([-25./12., 4., -3., 4./3., -1./4.])])
    if stencils == 7:
        forward_coe = np.transpose([np.array([-49./20., 6., -15./2., 20./3., -15./4., 6./5., -1./6.])])
    if stencils == 9: #used wikipedia's online calculator...
        forward_coe = np.transpose([np.array([-1.0187546202093276e+83, 2.9986995390864937e+83,
                                                -5.247724193324426e+83, 6.996965591015617e+83,
                                              -6.559655241526099e+83, 4.1981793545536496e+83,
                                              -1.7492413977236302e+83, 4.2838564842079276e+82,
                                              -4.685468029591042e+81])]) / 3.748374423901926e+82
        # forward_coe = np.transpose([np.array([15/7, 8, -14, 56/3., -35./2., 56./5., -14./3., 8./7., -1./8])])
    backward_coe = -forward_coe
    return np.concatenate([np.sum(forward * forward_coe, axis=0), np.sum(backward * backward_coe, axis=0)], axis=0) / h

