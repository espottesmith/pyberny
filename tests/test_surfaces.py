import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
                                  "points": [[0,0], [0.75, -0.75], [-1.75, 0.75], [0.5, 1.5], [0.5, 0.5]]},
                 "i_surface": {"function": i_surface,
                               "points": [[0, 0], [-1.0, -0.25], [1.0, 0.25], [0, -4.0], [-2.5, 0.5]]},
                 "v_surface": {"function": v_surface,
                               "points": [[0, 0], [2.0, 0], [0, 2], [1.5, 0.5], [-0.75, 4], [0.75, 1.0], [1.0, 0.75]]},
                 "t_surface": {"function": t_surface,
                               "points": [[0, 0], [-0.25, -1.25], [0.3, -2], [0, 0.2], [-0.5, 1.25], [4.0, 2.5]]},
                 "h_surface": {"function": h_surface,
                               "points": [[0, 0], [0.1, 0.1], [-0.5, -0.5], [-2.0, 1.25], [-3.0, -1.75], [-2.0, -1.0]]},
                 "halgren_lipscomb": {"function": halgren_lipscomb,
                                      "points": [[0.5, 0.5], [0.75, 3.0], [2.0, 2.0], [2.5, 3.75], [0.5, 4.0], [3.25, 1.75]]},
                 "cerjan_miller": {"function": cerjan_miller,
                                   "points": [[0, 0], [-1.0, 0], [-0.75, 1.0], [0.5, -0.6], [0.4, 0.2]]},
                 "quapp_wolfe_schlegel": {"function": quapp_wolfe_schlegel,
                                          "points": [[0, 0], [-1.5, -0.25], [-1.75, 0.5], [1.25, 0.25], [2.0, 2.0], [-2, -1.5]]},
                 "culot_dive_nguyen_ghuysen": {"function": culot_dive_nguyen_ghuysen,
                                               "points": [[0, 0], [1, -2.5], [-2.5, -1.75], [1.5, 1.25], [0, 1.75], [-2.0, 2.75]]}
                 }


def test_point(funct, point, prefix, setup=True):
    initial_point = np.array(point)

    if setup:
        relaxed = optimize2D(Berny2D(initial_point, debug=True, trust=0.03), Solver2D(funct), trajectory="trajectories/{}_{}_{}".format(prefix, point[0], point[1]))
    else:
        relaxed = optimize2D(Berny2D(initial_point, debug=True, trust=0.03), Solver2D(funct), trajectory="trajectories/{}_{}_{}".format(prefix, point[0], point[1]))

    traj = parse_trajectory("trajectories/{}_{}_{}".format(prefix, point[0], point[1]))

    xs = [e[0] for e in traj["geometries"]]
    ys = [e[1] for e in traj["geometries"]]
    zs = [e for e in traj["energies"]]
    trajectory = list(zip(xs, ys, zs))

    # visualize_trajectory(funct, trajectory)

    return relaxed


def setup_trajectories():
    for fstring, params in preset_points.items():
        for point in params["points"]:
            test_point(params["function"], point, fstring, setup=True)


setup_trajectories()