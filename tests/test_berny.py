# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys

import numpy as np
from numpy.linalg import LinAlgError
from numpy import dot, eye
from numpy.linalg import norm
from scipy.linalg import eigh

from berny.berny import (defaults, Point, update_hessian_min, update_hessian_ts,
                         update_trust, linear_search, quadratic_step_min,
                         quadratic_step_ts, is_converged, TrustRadiusException,
                         NoRootException)
from berny.Logger import Logger
from berny import Math

if sys.version_info[:2] >= (3, 5):
    from collections.abc import Generator
else:
    from berny._py2 import Generator


class State2D:
    """
    Object holding the current state of the optimizer on a 2D potential energy
    surface.

    :param np.ndarray geom: current geometry
    :param float trust: current trust radius
    :param np.ndarray hessian: current guess for the Hessian matrix
    :param Point future: coordinates for the next step in the optimization
    :param dict params: optimization parameters at the current step
    :param bool first: Is this the first step in the optimization
    :param Point previous: coordinates for the previous step in the optimization
    :param Point interpolated: point obtained by linear interpolation (for
           minimization only)
    :param float time: Current time (should typically be based on time.time()).
           Default is None.
    """


class Berny2D(Generator):
    """
    Generator that receives energy and gradients and yields the next geometry.
    This class is designed to be used to test new features using simple
    2-dimensional potential energy surfaces.

    :param np.ndarray geom: geometry to start with
    :param Logger log: used for logging if given
    :param bool debug: if True, the generator yields debug info on receiving
        the energy and gradients, otherwise it yields None
    :param dict restart: start from a state saved from previous run
        using ``debug=True``
    :param int maxsteps: abort after maximum number of steps
    :param int verbosity: if present and log is None, specifies the verbosity of
        the default :py:class:`~berny.Logger`
    :param bool transition_state: if True (default False), use a transition state
        search instead of a geometry optimization
    :param params: parameters that override the :py:data:`~berny.berny.defaults`

    The Berny2D object is to be used as follows::

        optimizer = Berny2D(geom)
        for geom in optimizer:
            # calculate energy and gradients (as vector)
            debug = optimizer.send((energy, gradients))
    """

