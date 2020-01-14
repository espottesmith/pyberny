import os
import shutil
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sympy

from utils.berny2d import Berny2D
from utils.optimizer2d import optimize2D, parse_trajectory
from utils.solver2d import Solver2D
from utils.surfaces2d import (muller_brown, i_surface, v_surface, t_surface,
                               h_surface, halgren_lipscomb, cerjan_miller,
                               adams, hoffman_noff_ruedenberg,
                               quapp_wolfe_schlegel, culot_dive_nguyen_ghuysen,
                               bifurcation, whirlpool, slot, visualize_surface,
                               visualize_trajectory)


preset_points = {"muller_brown": {"function": muller_brown,
                                  "points": [[0,0], [0.75, -0.75], [-1.75, 0.75], [0.5, 1.5], [0.5, 0.5]],
                                  "xrange": (-1.25, 1.25),
                                  "yrange": (-1.5, 1.75)},
                 "i_surface": {"function": i_surface,
                               "points": [[0, 0], [-1.0, -0.25], [1.0, 0.25], [0, -4.0], [-2.5, 0.5]],
                               "xrange": (-3, 3),
                               "yrange": (-1, 1)},
                 "v_surface": {"function": v_surface,
                               "points": [[0, 0], [2.0, 0], [0, 2], [1.5, 0.5], [-0.75, 4], [0.75, 1.0], [1.0, 0.75]],
                               "xrange": (-1, 4),
                               "yrange": (-1, 4)},
                 "t_surface": {"function": t_surface,
                               "points": [[0, 0], [-0.25, -1.25], [0.3, -2], [0, 0.2], [-0.5, 1.25], [4.0, 2.5]],
                               "xrange": (-3, 5),
                               "yrange": (-2, 3)},
                 "h_surface": {"function": h_surface,
                               "points": [[0, 0], [0.1, 0.1], [-0.5, -0.5], [-2.0, 1.25], [-3.0, -1.75], [-2.0, -1.0]],
                               "xrange": (-4, 4),
                               "yrange": (-2, 2)},
                 "halgren_lipscomb": {"function": halgren_lipscomb,
                                      "points": [[0.5, 0.5], [0.75, 3.0], [2.0, 2.0], [2.5, 3.75], [0.5, 4.0], [3.25, 1.75]],
                                      "xrange": (0.5, 4),
                                      "yrange": (0.5, 4)},
                 "cerjan_miller": {"function": cerjan_miller,
                                   "points": [[0, 0], [-1.0, 0], [-0.75, 1.0], [0.5, -0.6], [0.4, 0.2]],
                                   "xrange": (-2.5, 2.5),
                                   "yrange": (-1.5, 1.5)},
                 "quapp_wolfe_schlegel": {"function": quapp_wolfe_schlegel,
                                          "points": [[0, 0], [-1.5, -0.25], [-1.75, 0.5], [1.25, 0.25], [2.0, 2.0], [-2, -1.5]],
                                          "xrange": (-2, 2),
                                          "yrange": (-2, 2)},
                 "culot_dive_nguyen_ghuysen": {"function": culot_dive_nguyen_ghuysen,
                                               "points": [[0, 0], [1, -2.5], [-2.5, -1.75], [1.5, 1.25], [0, 1.75], [-2.0, 2.75]],
                                               "xrange": (-4.5, 4.5),
                                               "yrange": (-4.5, 4.5)}
                 }


def optimize_from_point(funct, point, prefix, setup=True):
    initial_point = np.array(point)

    if setup:
        relaxed = optimize2D(Berny2D(initial_point, debug=True, trust=0.03), Solver2D(funct), trajectory="trajectories/{}_{}_{}".format(prefix, point[0], point[1]))
    else:
        relaxed = optimize2D(Berny2D(initial_point, debug=True, trust=0.03), Solver2D(funct))

    return relaxed


def setup_trajectories():
    for fstring, params in preset_points.items():
        for point in params["points"]:
            optimize_from_point(params["function"], point, fstring, setup=True)


def test_preset_points():
    for fstring, params in preset_points.items():
        for point in params["points"]:
            trajectory = parse_trajectory("trajectories/{}_{}_{}".format(fstring, point[0], point[1]))
            try:
                relaxed = optimize2D(Berny2D(point, debug=True, trust=0.03), Solver2D(params["function"]), trajectory="temp_traj")
                for i, e in enumerate(list(relaxed)):
                    assert e == trajectory["geometries"][-1][i]
            finally:
                if os.path.exists("temp_traj"):
                    os.remove("temp_traj")


def test_random_points():
    for params in preset_points.values():
        xset = np.linspace(params["xrange"][0], params["xrange"][1], 1000000)
        yset = np.linspace(params["yrange"][0], params["yrange"][1], 1000000)

        xsample = random.sample(list(xset), 5)
        ysample = random.sample(list(yset), 5)

        for point in zip(xsample, ysample):
            geom = optimize_from_point(params["function"], point, None, setup=False)

            x, y = sympy.symbols('x y')
            vars = [x, y]
            funct = params["function"](x, y, sym=True)

            gradients = list()
            for e, var in enumerate(vars):
                grad = funct.diff(var)
                gradients.append(grad.subs([(x, geom[0]), (y, geom[1])]))

            for element in gradients:
                assert abs(element) < 0.45e-3


setup_trajectories()