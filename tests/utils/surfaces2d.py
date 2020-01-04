# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sympy


def muller_brown(x, y, *args, **kwargs):
    a = [-200, -100, -170, 15]
    b = [-1, -1, -6.5, 0.7]
    c = [0, 0, 11, 0.6]
    d = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    total = 0
    for i in range(4):
        total += a[i] * np.exp(b[i] * (x - x0[i]) ** 2 + c[i] * (x - x0[i]) * (y - y0[i]) + d[i] * (y - y0[i]) ** 2)

    return total


def pes_1_schlegel(x, y, *args, **kwargs):
    # Doesn't work; typo?
    a = [1.7, 1.7, 0.8, 0.8, -1, -1, -0.25, -0.25, -0.5]
    b = [1.5, 1.5, 4, 4, 14, 1, 4, 4, 4]
    c = [4, 4, 4, 4, 4, 4, 4, 4, 4]
    x0 = [-0.25, 1.75, -0.1, 1.6, 0.35, 1.15, -0.75, 2.25, 0.75]
    y0 = [0.5, 0.5, -0.9, 5, -0.75, -0.75, -0.75, -0.75, 1.2]

    total = 0
    for i in range(9):
        total += a[i] * np.exp(b[i] * (x - x0[i]) ** 2 + c[i] * (y - y0[i]) ** 2)

    return total


def letter_surface(v11, v22, v12, x, y):
    v11v = v11(x, y)
    v22v = v22(x, y)
    v12v = v12(x, y)

    v = (v11v + v22v)/2 - np.sqrt(((v11v + v22v)/2)**2 + v12v**2)

    return v


def i_surface(x, y, *args, **kwargs):
    def v11(x, y):
        return -0.1 * np.exp(-0.3 * (x + 2) ** 2 - 5 * y ** 2)

    def v22(x, y):
        return -0.1 * np.exp(-0.3 * (x - 2) ** 2 - 5 * y ** 2)

    def v12(x, y):
        return 0.02 * np.exp(-(x ** 2 + y ** 2))

    return letter_surface(v11, v22, v12, x, y)


def v_surface(x, y, *args, **kwargs):
    def v11(x, y):
        return -0.1 * np.exp(-0.4 * (x - 2) ** 2 - 5 * y ** 2)

    def v22(x, y):
        return -0.1 * np.exp(-5 * x ** 2 - 0.4 * (y - 2) ** 2)

    def v12(x, y):
        return 0.05 * np.exp(-0.4 * ((x - 0.2) ** 2 + (y - 0.8) ** 2))

    return letter_surface(v11, v22, v12, x, y)


def t_surface(x, y, *args, **kwargs):
    def v11(x, y):
        return -0.1 * np.exp(-0.1 * (x - 2) ** 2 - 5 * (y - 1.5) ** 2)

    def v22(x, y):
        return -0.05 * np.exp(-5 * x ** 2 - 0.3 * (y + 1) ** 2)

    def v12(x, y):
        return 0.02 * np.exp(-0.8 * (x ** 2 + (y - 1.5) ** 2))

    return letter_surface(v11, v22, v12, x, y)


def h_surface(x, y, *args, **kwargs):
    def v11(x, y):
        return -0.1 * np.exp(-0.12 * (x - 2) ** 2 - 3 * (y - 1) ** 2)

    def v22(x, y):
        return -0.1 * np.exp(-0.12 * (x + 2) ** 2 - 3 * (y + 1) ** 2)

    def v12(x, y):
        return 0.046 * np.exp(-0.5 * (x ** 2 + y ** 2))

    return letter_surface(v11, v22, v12, x, y)


def halgren_lipscomb(x, y, *args, **kwargs):
    return ((x - y) ** 2 - (5/3)**2) ** 2 + 4 * (x * y - 4) ** 2 + x - y


def cerjan_miller(x, y, *args, **kwargs):
    return (1 - y ** 2) * x ** 2 * np.exp(-x**2) + 0.5 * y ** 2


def adams(x, y, *args, **kwargs):
    return 2 * x ** 2 * (4 - x) + y ** 2 * (4 + y) - x * y * (6 - 17 * np.exp(-0.25 * (x ** 2 + y ** 2)))


def hoffman_noff_ruedenberg(x, y, *args, **kwargs):
    return (x * y ** 2 - y * x ** 2 + x **2 + 2 * y - 3) / 2


def quapp_wolfe_schlegel(x, y, *args, **kwargs):
    return x ** 4 + y ** 4 - 2 * x ** 2 - 4 * y ** 2 + x * y + 0.3 * x + 0.1 * y


def culot_drive_nguyen_ghuysen(x, y, *args, **kwargs):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def dey_janicki_ayers(x, y, *args, **kwargs):
    # not working
    a = [0.6, -2, -2, -2, -2]
    b = [1, 0.3, 1, 0.4, 1]
    c = [1, 0.4, 1, 1, 0.1]
    x0 = [0.1, 1.3, -1.5, 1.4, -1.3]
    y0 = [0.1, -1.6, -1.7, 1.8, 1.23]

    total = 0.5
    for i in range(5):
        total += a[i] * np.exp(b[i] * (x - x0[i]) ** 2 + c[i] * (y - y0[i]) ** 2)

    return total


def bifurcation(x, y, *args, **kwargs):
    return (2 - np.cos(np.pi * x)) * np.sin(np.pi * y / 2)


def serpentine(x, y, *args, **kwargs):
    # Some problem at low x range
    return np.arctan(-1 / (np.exp(y) * 1 / np.tan(x / 2 - np.pi / 4))) - 2 * np.exp((y - np.sin(x)) ** 2 / -2)


def whirlpool(x, y, *args, **kwargs):
    r = np.sqrt(x ** 2 + y ** 2)
    sig = np.arcsin(-2/5)
    return 1/2 * (1 - x / r * np.cos(np.log(r) + sig) - y / r * np.sin(np.log(r) + sig)) + 1/2 * np.log(r)


def slot(x, y, *args, **kwargs):
    return 2 / (np.exp(4 * x) + 1) - 0.2 * x - np.exp(-1 * (4 - x) * (y - 0.2 * (1 - x) * np.sin(2 * x)) ** 2)


def visualize_surface(f, xmin=-1, xmax=1, ymin=-1, ymax=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    xs, ys = np.meshgrid(x, y)
    zs = f(xs, ys)

    ax.plot_surface(xs, ys, zs, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('E')

    plt.show()


def visualize_trajectory(f, trajectory, xmin=-1, xmax=1, ymin=-1, ymax=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    xs, ys = np.meshgrid(x, y)
    zs = f(xs, ys)

    ax.plot_surface(xs, ys, zs, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('E')

    t_xs = np.array([t[0] for t in trajectory])
    t_ys = np.array([t[1] for t in trajectory])
    t_zs = np.array([t[2] for t in trajectory])
    ax.plot(t_xs, t_ys, t_zs, c='k')

    plt.show()
