import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from utils.berny2d import Berny2D
from utils.optimizer2d import optimize2D, parse_trajectory
from utils.solver2d import Solver2D
from utils.surfaces2d import (muller_brown, i_surface, v_surface, t_surface,
                               h_surface, halgren_lipscomb, cerjan_miller,
                               adams, hoffman_noff_ruedenberg,
                               quapp_wolfe_schlegel, culot_drive_nguyen_ghuysen,
                               bifurcation, whirlpool, slot, visualize_surface,
                               visualize_trajectory)


def test_basic():
    initial_point = np.array([0, 0])
    f = muller_brown

    relaxed = optimize2D(Berny2D(initial_point, debug=True, trust=0.03), Solver2D(muller_brown), trajectory="trajectory")

    traj = parse_trajectory("trajectory")

    xs = [e[0] for e in traj["geometries"]]
    ys = [e[1] for e in traj["geometries"]]
    zs = [e for e in traj["energies"]]
    trajectory = list(zip(xs, ys, zs))

    visualize_trajectory(muller_brown, trajectory)


test_basic()