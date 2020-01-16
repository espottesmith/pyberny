import os

import numpy as np
from pkg_resources import resource_filename

import pytest
from pytest import approx

from berny import Berny, optimize, geomlib
from berny.solvers import MopacSolver

this_file = os.path.abspath(__file__)
this_dir = os.path.dirname(__file__)

@pytest.fixture
def mopac(scope='session'):
    return MopacSolver()


@pytest.fixture
def ethanol():
    return geomlib.readfile(resource_filename('tests', 'ethanol.xyz'))


@pytest.fixture
def aniline():
    return geomlib.readfile(resource_filename('tests', 'aniline.xyz'))


def test_ethanol(mopac, ethanol):
    berny = Berny(ethanol, steprms=0.01, stepmax=0.05, maxsteps=5)
    final = optimize(berny, mopac)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([14.95, 52.58, 61.10], rel=1e-3)


def test_aniline(mopac, aniline):
    berny = Berny(aniline, steprms=0.01, stepmax=0.05, maxsteps=8)
    final = optimize(berny, mopac)
    inertia_princpl = np.linalg.eigvalsh(final.inertia)
    assert inertia_princpl == approx([90.94, 193.1, 283.9], rel=1e-3)


def mass_optimize_ground_state():
    test_set = os.path.join(this_dir, "test_sets", "ground_state")

    steps = {}

    for file in os.listdir(test_set):
        print()
        this_step = 0
        init_geom = geomlib.readfile(os.path.join(test_set, file))
        berny = Berny(init_geom, debug=True, maxsteps=500)

        mopac = MopacSolver(cmd="/opt/mopac/MOPAC2016.exe", workdir="/Users/ewcss/software/jupyter/")

        next(mopac)
        for geom in berny:
            print(file)
            energy, gradients = mopac.send((list(geom), geom.lattice))
            berny.send((energy, gradients))
            this_step += 1

        steps[file.split(".")[0]] = this_step

    print(steps)
    print("Total steps: {}".format(sum(steps.values())))