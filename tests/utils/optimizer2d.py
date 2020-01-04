# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


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
    try:
        next(solver)
        for geom in optimizer:
            energy, gradients = solver.send(list(geom))
            if trajectory:
                geom.dump(trajectory, 'xyz')
            optimizer.send((energy, gradients))
    finally:
        if trajectory:
            trajectory.close()
    return geom