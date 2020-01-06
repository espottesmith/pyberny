# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np


def optimize2D(optimizer, solver, trajectory=None):
    """
    Optimize a geometry on a 2D potential energy surface with respect to a
    solver.

    :param generator optimizer: Optimizer object with the same generator interface
        as :py:func:`tests.Berny2D`
    :param generator solver: unprimed generator that receives a geometry as an
        np.ndarray and yields the energy and gradients (as a vector)
    :param trajectory: filename for the XYZ trajectory

    Returns the optimized geometry.
    The function is equivalent to::
        next(solver)
        for geom in optimizer:
            energy, gradients = solver.send((list(geom), geom.lattice))
            optimizer.send((energy, gradients))
    """

    if trajectory:
        trajectory = open(trajectory, 'w')
        trajectory.write("x\ty\tEnergy\tgrad_x\tgrad_y\n")
    try:
        next(solver)
        for geom in optimizer:
            energy, gradients = solver.send(list(geom))
            if trajectory:
                trajectory.write("{}\t{}\t{}\t{}\t{}\n".format(geom[0], geom[1], energy, gradients[0], gradients[1]))
            optimizer.send((energy, gradients))
    finally:
        if trajectory:
            trajectory.close()
    return geom


def parse_trajectory(trajectory):
    """
    From a trajectory output (as defined in optimize2D), parse to get the
    geometries, energies, and gradients at each point.

    :param str trajectory: filename referring to an optimization trajectory

    :return: dict
    """

    with open(trajectory, 'r') as file:
        contents = file.readlines()

        traj = {"geometries": list(),
                "energies": list(),
                "gradients": list()}

        for line in contents[1:]:
            c = line.split("\t")
            traj["geometries"].append(np.array([float(c[0]), float(c[1])]))
            traj["energies"].append(float(c[2]))
            traj["gradients"].append(np.array([float(c[3]), float(c[4])]))

        return traj
