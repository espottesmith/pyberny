# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import sys
from collections import namedtuple
from itertools import chain

import numpy as np
from numpy import dot, eye
from numpy.linalg import norm, matrix_power
from scipy.optimize import brute

from . import Math
from .coords import InternalCoords
from .Logger import Logger

if sys.version_info[:2] >= (3, 5):
    from collections.abc import Generator
else:
    from ._py2 import Generator  # noqa

__version__ = '0.2.1'

defaults = {
    'gradientmax': 0.45e-3,
    'gradientrms': 0.3e-3,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
    'trust': 0.3,
    'dihedral': True,
    'superweakdih': False,
}
"""
- gradientmax, gradientrms, stepmax, steprms:
    Convergence criteria in atomic units ("step" refers to the step in
    internal coordinates, assuming radian units for angles).

- trust:
    Initial trust radius in atomic units. It is the maximum RMS of the
    quadratic step (see below).

- dihedral:
    Form dihedral angles.

- superweakdih:
    Form dihedral angles containing two or more noncovalent bonds.
"""


Point = namedtuple('Point', 'q E g')


class State(object):
    """
    Object holding the current state of the optimizer.

    :param Geometry geom: current geometry
    :param InternalCoords coords: current internal coordinates
    :param float trust: current trust radius
    :param np.ndarray hessian: current guess for the Hessian matrix
    :param np.narray weights: current coordinate weights
    :param Point future: coordinates for the next step in the optimization
    :param dict params: optimization parameters at the current step
    :param bool first: Is this the first step in the optimization
    """

    def __init__(self, geom, coords, trust, hessian, weights, future, params,
                 first=True):
        self.geom = geom
        self.coords = coords
        self.trust = trust
        self.H = hessian
        self.weights = weights
        self.future = future
        self.params = params
        self.first = first


class Berny(Generator):
    """
    Generator that receives energy and gradients and yields the next geometry.

    :param Geometry geom: geometry to start with
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

    The Berny object is to be used as follows::

        optimizer = Berny(geom)
        for geom in optimizer:
            # calculate energy and gradients (as N-by-3 matrix)
            debug = optimizer.send((energy, gradients))
    """

    def __init__(self, geom, log=None, debug=False, restart=None, maxsteps=100,
                 verbosity=None, transition_state=False, **params):
        self._log = log or Logger(verbosity=verbosity or 0)
        self._debug = debug
        self._maxsteps = maxsteps
        self._converged = False
        self._n = 0
        params = dict(chain(defaults.items(), params.items()))
        coords = InternalCoords(geom,
            dihedral=params['dihedral'],
            superweakdih=params['superweakdih'],
        )
        s = self._state = State(geom=geom, coords=coords, trust=params["trust"],
                                hessian=coords.hessian_guess(geom),
                                weights=coords.weights(geom),
                                future=Point(coords.eval_geom(geom),
                                             None, None),
                                params=params,
                                first=True)

        self.transition_state = transition_state
        if restart:
            vars(s).update(restart)
            return
        for line in str(s.coords).split('\n'):
            self._log(line)

    def __next__(self):
        assert self._n <= self._maxsteps
        if self._n == self._maxsteps or self._converged:
            raise StopIteration
        self._n += 1
        return self._state.geom

    def send(self, energy_gradients):
        log = self._log
        log.n = self._n
        s = self._state
        energy, gradients = energy_gradients
        gradients = np.array(gradients)
        log('Energy: {:.12}'.format(energy), level=1)
        B = s.coords.B_matrix(s.geom)
        B_inv = B.T.dot(Math.pinv(np.dot(B, B.T), log=log))
        current = Point(
            s.future.q, energy, dot(B_inv.T, gradients.reshape(-1))
        )
        if not s.first:
            if self.transition_state:
                s.H = update_hessian_ts(
                    s.H, current.q-s.best.q, current.g-s.best.g, log=log
                )
            else:
                s.H = update_hessian_min(
                    s.H, current.q-s.best.q, current.g-s.best.g, log=log
                )
            s.trust = update_trust(
                s.trust,
                current.E-s.previous.E,
                s.predicted.E-s.interpolated.E,
                s.predicted.q-s.interpolated.q,
                log=log
            )
            if self.transition_state:
                # Should there be some alternate method used here?
                s.interpolated = current
            else:
                dq = s.best.q-current.q
                t, E = linear_search(
                    current.E, s.best.E, dot(current.g, dq), dot(s.best.g, dq),
                    log=log
                )
                s.interpolated = Point(
                    current.q+t*dq, E, t*s.best.g+(1-t)*current.g
                )
        else:
            s.interpolated = current
        if s.trust < 1e-6:
            raise TrustRadiusException('The trust radius got too small; check forces?')
        proj = dot(B, B_inv)
        H_proj = proj.dot(s.H).dot(proj) + 1000*(eye(len(s.coords))-proj)
        if self.transition_state:
            dq, dE, on_sphere = quadratic_step_ts_basic(
                dot(proj, s.interpolated.g), H_proj, s.trust, log=log
            )
        else:
            dq, dE, on_sphere = quadratic_step_min(
                dot(proj, s.interpolated.g), H_proj, s.trust, log=log
            )
        s.predicted = Point(s.interpolated.q+dq, s.interpolated.E+dE, None)
        dq = s.predicted.q-current.q
        log('Total step: RMS: {:.3}, max: {:.3}'.format(
            Math.rms(dq), max(abs(dq))
        ))
        q, s.geom = s.coords.update_geom(
            s.geom, current.q, s.predicted.q-current.q, B_inv, log=log
        )
        s.future = Point(q, None, None)
        s.previous = current
        if s.first or current.E < s.best.E:
            s.best = current
        s.first = False
        self._converged = is_converged(
            gradients, s.future.q-current.q, on_sphere, s.params, log=log
        )
        if self._n == self._maxsteps:
            log('Maximum number of steps reached')
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
        return Generator.close(self, *args, **kwargs)


def no_log(msg, **kwargs):
    pass


def update_hessian_min(H, dq, dg, log=no_log):
    dH = dg[None, :]*dg[:, None]/dot(dq, dg) - \
        H.dot(dq[None, :]*dq[:, None]).dot(H)/dq.dot(H).dot(dq)  # BFGS update
    log('Hessian update information:')
    log('* Change: RMS: {:.3}, max: {:.3}'.format(Math.rms(dH), abs(dH).max()))
    return H+dH


def update_hessian_ts(H, dq, dg, log=no_log):
    # Uses the Bofill method
    # Mix of Powell-symmetric Broyden (PSB) and SR1
    # delta(H)_bofill = phi delta(H)_SR1 + (1 - phi) delta(H)_PSB
    # phi = [(delta(g) - H_old delta(x))T delta(x)]^2/[|delta(g) - H_old delta(x)|^2 |delta(x)|^2]
    quad_err = dg - H.dot(dq)
    qtq = dot(dq, dq)

    phi = dot(quad_err, dq) ** 2 / (norm(quad_err)**2 * norm(dq)**2)
    dH_SR1 = (quad_err[None, :] * quad_err[:, None]) / (dot(dq, quad_err))
    dH_PSB = ((quad_err[None, :] * dq[:, None]) + (dq[None, :] * quad_err[:, None])) / qtq - \
             dot(dq, quad_err) * (dq[None, :] * dq[:, None]) / qtq ** 2

    dH = phi * dH_SR1 + (1 - phi) * dH_PSB
    log('Hessian update information:')
    log('* Change: RMS {:.3}, max {:.3}'.format(Math.rms(dH), abs(dH).max()))
    return H + dH


def update_trust(trust, dE, dE_predicted, dq, log=no_log):
    if dE != 0:
        r = dE/dE_predicted  # Fletcher's parameter
    else:
        r = 1.
    log("Trust update: Fletcher's parameter: {:.3}".format(r))
    if r < 0.25:
        return norm(dq)/4
    elif r > 0.75 and abs(norm(dq)-trust) < 1e-10:
        return 2*trust
    else:
        return trust


def linear_search(E0, E1, g0, g1, log=no_log):
    log('Linear interpolation:')
    log('* Energies: {:.8}, {:.8}'.format(E0, E1))
    log('* Derivatives: {:.3}, {:.3}'.format(g0, g1))
    t, E = Math.fit_quartic(E0, E1, g0, g1)
    if t is None or t < -1 or t > 2:
        t, E = Math.fit_cubic(E0, E1, g0, g1)
        if t is None or t < 0 or t > 1:
            if E0 <= E1:
                log('* No fit succeeded, staying in new point')
                return 0, E0

            else:
                log('* No fit succeeded, returning to best point')
                return 1, E1
        else:
            msg = 'Cubic interpolation was performed'
    else:
        msg = 'Quartic interpolation was performed'
    log('* {}: t = {:.3}'.format(msg, t))
    log('* Interpolated energy: {:.8}'.format(E))
    return t, E


def quadratic_step_min(g, H, trust, log=no_log):
    ev = np.linalg.eigvalsh((H+H.T)/2)
    rfo = np.vstack((np.hstack((H, g[:, None])),
                     np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo+rfo.T)/2)
    dq = V[:-1, 0]/V[-1, 0]
    l = D[0]
    if norm(dq) <= trust:
        log('Pure RFO step was performed:')
        on_sphere = False
    else:
        def steplength(l):
            return norm(np.linalg.solve(l*eye(H.shape[0])-H, g))-trust
        l = Math.findroot(steplength, ev[0])  # minimization on sphere
        dq = np.linalg.solve(l*eye(H.shape[0])-H, g)
        on_sphere = True
        log('Minimization on sphere was performed:')
    dE = dot(g, dq)+0.5*dq.dot(H).dot(dq)  # predicted energy change
    log('* Trust radius: {:.2}'.format(trust))
    log('* Number of negative eigenvalues: {}'.format((ev < 0).sum()))
    log('* Lowest eigenvalue: {:.3}'.format(ev[0]))
    log('* lambda: {:.3}'.format(l))
    log('Quadratic step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
    log('* Predicted energy change: {:.3}'.format(dE))
    return dq, dE, on_sphere


def quadratic_step_ts_basic(g, H, trust, log=no_log):
    ev = np.linalg.eigvalsh((H+H.T)/2)
    rfo = np.vstack((np.hstack((H, g[:, None])),
                     np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo+rfo.T)/2)
    dq = V[:-1, 1]/V[-1, 1]
    l = D[1]
    if norm(dq) <= trust:
        log('Pure RFO step was performed:')
        on_sphere = False
    else:
        def abs_steplength(l):
            return abs(norm(np.linalg.solve(l*eye(H.shape[0])-H, g))-trust)
        root = brute(abs_steplength, [(ev[0], ev[1])])
        l = root[0]
        if l > 1e-5:
            raise NoRootException("No root found between the first and second "
                                  "eigenvalues; quadratic step failed.")
        dq = np.linalg.solve(l*eye(H.shape[0])-H, g)
        on_sphere = True
        log('Minimization on sphere was performed:')
    dE = dot(g, dq)+0.5*dq.dot(H).dot(dq)  # predicted energy change
    log('* Trust radius: {:.2}'.format(trust))
    log('* Number of negative eigenvalues: {}'.format((ev < 0).sum()))
    log('* Lowest eigenvalue: {:.3}'.format(ev[0]))
    log('* lambda: {:.3}'.format(l))
    log('Quadratic step: RMS: {:.3}, max: {:.3}'.format(Math.rms(dq), max(abs(dq))))
    log('* Predicted energy change: {:.3}'.format(dE))
    return dq, dE, on_sphere


def quadratic_step_ts_partition(g, H, trust, log=no_log):
    pass


def is_converged(forces, step, on_sphere, params, log=no_log):
    criteria = [
        ('Gradient RMS', Math.rms(forces), params['gradientrms']),
        ('Gradient maximum', np.max(abs(forces)), params['gradientmax'])
    ]
    if on_sphere:
        criteria.append(('Minimization on sphere', False))
    else:
        criteria.extend([
            ('Step RMS', Math.rms(step), params['steprms']),
            ('Step maximum', np.max(abs(step)), params['stepmax'])
        ])
    log('Convergence criteria:')
    all_matched = True
    for crit in criteria:
        if len(crit) > 2:
            result = crit[1] < crit[2]
            msg = '{:.3} {} {:.3}'.format(crit[1], '<' if result else '>', crit[2])
        else:
            msg, result = crit
        msg = '{}: {}'.format(crit[0], msg) if msg else crit[0]
        msg = '* {} => {}'.format(msg, 'OK' if result else 'no')
        log(msg)
        if not result:
            all_matched = False
    if all_matched:
        log('* All criteria matched', level=1)
    return all_matched


class TrustRadiusException(Exception):
    pass


class NoRootException(Exception):
    pass