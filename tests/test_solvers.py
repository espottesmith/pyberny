# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/
from __future__ import division

import os
import tempfile
import subprocess
import shutil

import numpy as np
import sympy


def Solver2D(f, *args, **kwargs):
    geom = yield

    while True:
        energy = f(geom, *args, **kwargs)
        coords = np.array([c for c in geom])
        gradients = np.zeros(coords.shape)

        x, y = sympy.symbols('x y')
        vars = [x, y]
        funct = f(x, y)

        for var in enumerate(vars):
            grad = funct.diff(var)
            gradients.append(grad.subs([(x, coords[0]), (y, coords[1])]))

        geom = yield energy, gradients