# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
import time

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

    def __init__(self, geom, trust, hessian, future, params,
                 first=True, previous=None, interpolated=None, t=None):
        self.geom = geom
        self.trust = trust
        self.H = hessian
        self.future = future
        self.params = params
        self.first = first
        self.previous = previous
        self.interpolated = interpolated
        self.time = t or time.time()

    def as_dict(self):
        d = {"geom": self.geom,
             "trust": self.trust,
             "hessian": self.H,
             "future": self.future,
             "params": self.params,
             "first": self.first,
             "previous": self.previous,
             "interpolated": self.interpolated,
             "time": self.time}
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["geom"], d["trust"], d["hessian"], d["future"],
                   d["params"], first=d["first"], previous=d["previous"],
                   interpolated=d["interpolated"])


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

    def __init__(self, geom, log=None, debug=False, restart=None, maxsteps=100,
                 verbosity=None, transition_state=False, **params):
        self._log = log or Logger(verbosity=verbosity or 0)
        self._debug = debug
        self._maxsteps = maxsteps
        self._converged = False
        self._n = 0
        p = defaults
        p.update(params)

        if transition_state:
            hess = np.array([[-1, 0], [0, 1]])
        else:
            hess = np.array([[1, 0], [0, 1]])

        s = self._state = State2D(geom=geom, trust=p["trust"],
                                  hessian=hess,
                                  future=Point(geom, None, None),
                                  params=p,
                                  first=True)

        self.transition_state = transition_state
        if restart:
            vars(s).update(restart)
            return
        self._log(geom)

    def __next__(self):
        assert self._n <= self._maxsteps
        if self._n == self._maxsteps or self._converged:
            raise StopIteration
        self._n += 1
        return self._state.geom

    def update_hessian_exact(self, H):
        log = self._log
        log.n = self._n
        s = self._state
        # No need to convert from internal coordinates; we stay "Cartesian"
        s.H = H
        log("State Hessian updated from calculated exact Hessian")

    def send(self, energy_gradients):
        log = self._log
        log.n = self._n
        s = self._state
        energy, gradients = energy_gradients
        gradients = np.array(gradients)
        log('Energy: {:.12f}'.format(energy), level=1)

        current = Point(s.future.q, energy, gradients)
        # print("Current", current)

        if not s.first:
            if self.transition_state:
                s.H = update_hessian_ts(
                    s.H, current.q-s.best.q, current.g-s.best.g, log=log
                )
            else:
                s.H = update_hessian_min(
                    s.H, current.q - s.best.q, current.g - s.best.g, log=log
                )
            s.trust = update_trust(s.trust,
                                   current.E - s.previous.E,
                                   s.predicted.E - s.interpolated.E,
                                   s.predicted.q - s.interpolated.q,
                                   log=log)
            if self.transition_state:
                s.interpolated = current
            else:
                dq = s.best.q-current.q
                t, E = linear_search(
                    current.E, s.best.E, dot(current.g, dq), dot(s.best.g, dq),
                    log=log
                )
                s.interpolated = Point(
                    current.q + t * dq, E, t * s.best.g + (1 - t) * current.g
                )
        else:
            s.interpolated = current
        if s.trust < s.params["min_trust"]:
            if s.params["fail_low_trust"]:
                raise TrustRadiusException("The trust radius got too small, check forces?")
            else:
                s.trust = s.params["min_trust"]

        H_proj = s.H

        if self.transition_state:
            dq, dE, on_sphere = quadratic_step_ts(
                s.interpolated.g, H_proj, s.trust, log=log
            )
        else:
            dq, dE, on_sphere = quadratic_step_min(
                s.interpolated.g, H_proj, s.trust, log=log
            )
        s.predicted = Point(s.interpolated.q + dq, s.interpolated.E + dE, None)
        dq = s.predicted.q - current.q
        log('Total step: RMS: {:.3}, max: {:.3}'.format(
            Math.rms(dq), max(abs(dq))
        ))
        q = s.predicted.q
        s.geom = q
        s.future = Point(q, None, None)
        s.previous = current
        if s.first or current.E < s.best.E:
            s.best = current
        s.first = False
        self._converged = is_converged(
            gradients, s.future.q - current.q, on_sphere, s.params, log=log
        )
        if self._n == self._maxsteps:
            log('Maximum number of steps reached')

        current_time = time.time()
        log("Time of last step: {:10.2f}s".format(current_time - s.time))
        s.time = current_time
        if self._debug:
            return vars(s).copy()

    @property
    def converged(self):
        return self._converged

    @property
    def state(self):
        return self._state

    @property
    def maxsteps(self):
        return self._maxsteps

    def throw(self, *args, **kwargs):
        return Generator.throw(self, *args, **kwargs)