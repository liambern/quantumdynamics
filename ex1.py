import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# 1.


def grid(x_min, x_max, h_approx):
    number = int((x_max-x_min) / h_approx)
    h = (x_max-x_min) / (number - 1)
    grid = np.linspace(x_min, x_max, num=number)
    return grid, h


def d1(f, h, stencils, periodic=False):
    rolling = []
    for i in range(-stencils//2+1, stencils//2+1):
        rolling.append([np.roll(f, -i)])
    rolling = np.concatenate(rolling, axis=0)
    if stencils == 5:
        coefficients = np.transpose([np.array([1./12., -2./3., 0., 2./3., -1./12.])])
    if stencils == 7:
        coefficients = np.transpose([np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])])
    if stencils == 9:
        coefficients = np.transpose([np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])])
    if periodic:
        return np.sum(rolling * coefficients, axis=0) / h
    else:
        return (np.sum(rolling * coefficients, axis=0) / h)[stencils//2:-stencils//2]


def d2(f, h, stencils, periodic=False):
    rolling = []
    for i in range(-stencils//2+1, stencils//2+1):
        rolling.append([np.roll(f, -i)])
    rolling = np.concatenate(rolling, axis=0)
    if stencils == 5:
        coefficients = np.transpose([np.array([-1/12, 4/3, -5/2, 4/3, -1/12])])
    if stencils == 7:
        coefficients = np.transpose([np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])])
    if stencils == 9:
        coefficients = np.transpose([np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])])
    if periodic:
        return np.sum(rolling * coefficients, axis=0) / h**2.
    else:
        return (np.sum(rolling * coefficients, axis=0) / h**2.)[stencils//2:-stencils//2]
def f(x):
    return np.exp(x)

x, h= grid(-1., 1., 1.e-3)
s = 9
print(d2(f(x), h, s) - f(x)[s//2:-s//2])