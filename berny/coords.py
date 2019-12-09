# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import division

from collections import OrderedDict
from itertools import combinations, product

import numpy as np
from numpy import dot, pi
from numpy.linalg import norm

from . import Math
from .species_data import get_property

angstrom = 1/0.52917721092


class InternalCoord(object):
    def __init__(self, C=None):
        if C is not None:
            self.weak = sum(
                not C[self.idx[i], self.idx[i + 1]] for i in range(len(self.idx) - 1)
            )
        else:
            self.weak = 0

    def __eq__(self, other):
        return self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)

    def __repr__(self):
        args = list(map(str, self.idx))
        if self.weak is not None:
            args.append('weak=' + str(self.weak))
        return '{}({})'.format(self.__class__.__name__, ', '.join(args))


class Bond(InternalCoord):
    def __init__(self, i, j, **kwargs):
        if i > j:
            i, j = j, i
        self.i = i
        self.j = j
        self.idx = i, j
        InternalCoord.__init__(self, **kwargs)

    def hessian(self, rho):
        return 0.45*rho[self.i, self.j]

    def weight(self, rho, coords):
        return rho[self.i, self.j]

    def center(self, ijk):
        return np.round(ijk[[self.i, self.j]].sum(0))

    def eval(self, coords, grad=False, second=False):
        v = (coords[self.i] - coords[self.j]) * angstrom
        r = norm(v)
        if grad:
            return r, [v / r, -v / r]
        elif second:
            mat = np.zeros((6, 6))
            comp_coords = np.hstack([coords[self.i], coords[self.j]])
            for ii, i_coord in enumerate(comp_coords):
                for jj, j_coord in enumerate(comp_coords):
                    if jj <= ii:
                        vi = v[ii % 3]
                        vj = v[jj % 3]
                        # Is it the same atom?
                        ab = int(ii // 3 == jj // 3)
                        # Is it the same coordinate (x, y, z)?
                        ij = int(ii % 3 == jj % 3)
                        val = (-1) ** ab * (vi * vj - ij)/r
                        mat[ii, jj] = val
                        mat[jj, ii] = val
            return r, mat
        else:
            return r

    def as_dict(self):
        bond_dict = {"@module": self.__class__.__module__,
                     "@class": self.__class__.__name__,
                     "i": self.i,
                     "j": self.j,
                     "idx": self.idx,
                     "weak": self.weak}
        return bond_dict

    @classmethod
    def from_dict(cls, d):
        bond = cls(d["i"], d["j"])
        if d["weak"] is not None:
            bond.weak = d["weak"]
        return bond


class Angle(InternalCoord):
    def __init__(self, i, j, k, **kwargs):
        if i > k:
            i, j, k = k, j, i
        self.i = i
        self.j = j
        self.k = k
        self.idx = i, j, k
        InternalCoord.__init__(self, **kwargs)

    def hessian(self, rho):
        return 0.15 * (rho[self.i, self.j] * rho[self.j, self.k])

    def weight(self, rho, coords):
        f = 0.12
        return np.sqrt(rho[self.i, self.j] * rho[self.j, self.k]) * (
                f + (1 - f) * np.sin(self.eval(coords))
        )

    def center(self, ijk):
        return np.round(2 * ijk[self.j])

    def eval(self, coords, grad=False, second=False):
        v1 = (coords[self.i] - coords[self.j]) * angstrom
        v2 = (coords[self.k] - coords[self.j]) * angstrom
        dot_product = np.dot(v1, v2) / (norm(v1)*norm(v2))
        if dot_product < -1:
            dot_product = -1
        elif dot_product > 1:
            dot_product = 1
        phi = np.arccos(dot_product)
        if not grad and not second:
            return phi
        if abs(phi) > pi-1e-6:
            gradient = [
                (pi - phi) / (2 * norm(v1) ** 2) * v1,
                (1 / norm(v1) - 1 / norm(v2)) * (pi - phi) / (2 * norm(v1)) * v1,
                (pi - phi) / (2 * norm(v2) ** 2) * v2
            ]
        else:
            gradient = [
                1 / np.tan(phi) * v1 / norm(v1) ** 2
                - v2 / (norm(v1) * norm(v2) * np.sin(phi)),
                (v1 + v2) / (norm(v1) * norm(v2) * np.sin(phi))
                - 1 / np.tan(phi) * (v1 / norm(v1) ** 2 + v2 / norm(v2) ** 2),
                1 / np.tan(phi) * v2 / norm(v2) ** 2
                - v1 / (norm(v1) * norm(v2) * np.sin(phi))
            ]
        if grad:
            return phi, gradient
        elif second:
            if phi > pi-1e6:
                # Derivatives not well defined for linear angles
                return phi, np.zeros((9, 9))

            cosq = dot_product
            sinq = np.sqrt(1 - cosq**2)
            l1 = norm(v1)
            l2 = norm(v2)
            mat = np.zeros((9, 9))
            comp_coords = np.hstack([coords[self.i], coords[self.j], coords[self.k]])
            for ii, i_coord in enumerate(comp_coords):
                for jj, j_coord in enumerate(comp_coords):
                    if jj <= ii:
                        v1i = v1[ii % 3]
                        v2i = v2[jj % 3]
                        v1j = v1[jj % 3]
                        v2j = v2[jj % 3]
                        # Is it the same coordinate (x, y, z)?
                        ij = (ii % 3 == jj % 3)
                        term_1 = zeta(ii//3, 0, 1) * zeta(jj//3, 0, 1) * (v1i * v2j + v2i + v1j - 3 *
                                                                          v1i * v1j * cosq + ij * cosq) / l1**2
                        term_2 = zeta(ii//3, 2, 1) * zeta(jj//3, 2, 1) * (v1i * v2j + v2i + v1j - 3 *
                                                                          v2i * v2j * cosq + ij * cosq) / l2**2
                        term_3 = zeta(ii//3, 0, 1) * zeta(jj//3, 2, 1) * (v1i * v1j + v2i * v2j - v1i * v2j * cosq -
                                                                          ij) / (l1 * l2)
                        term_4 = zeta(ii//3, 2, 1) * zeta(jj//3, 0, 1) * (v1i * v1j + v2i * v2j - v2i * v1j * cosq -
                                                                          ij) / (l1 * l2)
                        term_5 = -1 * cosq * grad[ii // 3][ii % 3] * grad[jj // 3][jj % 3]
                        val = sum([term_1, term_2, term_3, term_4, term_5]) / sinq
                        mat[ii, jj] = val
                        mat[jj, ii] = val
            return phi, mat

    def as_dict(self):
        angle_dict = {"@module": self.__class__.__module__,
                      "@class": self.__class__.__name__,
                      "i": self.i,
                      "j": self.j,
                      "k": self.k,
                      "idx": self.idx,
                      "weak": self.weak}
        return angle_dict

    @classmethod
    def from_dict(cls, d):
        angle = cls(d["i"], d["j"], d["k"])
        if d["weak"] is not None:
            angle.weak = d["weak"]
        return angle


class Dihedral(InternalCoord):
    def __init__(self, i, j, k, l, weak=0, angles=None, C=None, **kwargs):
        if j > k:
            i, j, k, l = l, k, j, i
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.idx = (i, j, k, l)
        self.angles = angles
        self.weak = weak
        InternalCoord.__init__(self, **kwargs)

    def hessian(self, rho):
        return 0.005 * rho[self.i, self.j] * rho[self.j, self.k] * rho[self.k, self.l]

    def weight(self, rho, coords):
        f = 0.12
        th1 = Angle(self.i, self.j, self.k).eval(coords)
        th2 = Angle(self.j, self.k, self.l).eval(coords)
        return ((rho[self.i, self.j] * rho[self.j, self.k] * rho[self.k, self.l]) ** (1 / 3)
                * (f + (1 - f) * np.sin(th1))
                * (f + (1 - f) * np.sin(th2))
        )

    def center(self, ijk):
        return np.round(ijk[[self.j, self.k]].sum(0))

    def eval(self, coords, grad=False, second=False):
        v1 = (coords[self.i] - coords[self.j]) * angstrom
        v2 = (coords[self.l] - coords[self.k]) * angstrom
        w = (coords[self.k] - coords[self.j]) * angstrom
        ew = w / norm(w)
        a1 = v1 - dot(v1, ew) * ew
        a2 = v2 - dot(v2, ew) * ew
        sgn = np.sign(np.linalg.det(np.array([v2, v1, w])))
        sgn = sgn or 1
        dot_product = dot(a1, a2) / (norm(a1) * norm(a2))
        if dot_product < -1:
            dot_product = -1
        elif dot_product > 1:
            dot_product = 1
        phi = np.arccos(dot_product) * sgn
        if grad:
            if abs(phi) > pi-1e-6:
                g = Math.cross(w, a1)
                g = g / norm(g)
                A = dot(v1, ew) / norm(w)
                B = dot(v2, ew) / norm(w)
                grad = [
                    g / (norm(g) * norm(a1)),
                    -((1 - A) / norm(a1) - B / norm(a2)) * g,
                    -((1 + B) / norm(a2) + A / norm(a1)) * g,
                    g / (norm(g) * norm(a2))
                ]
            elif abs(phi) < 1e-6:
                g = Math.cross(w, a1)
                g = g/norm(g)
                A = dot(v1, ew) / norm(w)
                B = dot(v2, ew) / norm(w)
                grad = [
                    g / (norm(g) * norm(a1)),
                    -((1 - A) / norm(a1) + B / norm(a2)) * g,
                    ((1+B) / norm(a2) - A / norm(a1)) * g,
                    -g / (norm(g) * norm(a2))
                ]
            else:
                A = dot(v1, ew) / norm(w)
                B = dot(v2, ew) / norm(w)
                grad = [
                    1 / np.tan(phi) * a1 / norm(a1) ** 2 - a2 / (norm(a1) * norm(a2) * np.sin(phi)),
                    ((1 - A) * a2 - B * a1) / (norm(a1) * norm(a2) * np.sin(phi)) -
                    1 / np.tan(phi) * ((1 - A) * a1 / norm(a1) ** 2 - B * a2 / norm(a2) ** 2),
                    ((1 + B) * a1 + A * a2) / (norm(a1) * norm(a2) * np.sin(phi)) -
                    1 / np.tan(phi) * ((1 + B) * a2 / norm(a2) ** 2 + A * a1 / norm(a1) ** 2),
                    1 / np.tan(phi) * a2 / norm(a2) ** 2 - a1 / (norm(a1) * norm(a2) * np.sin(phi))
                ]
            return phi, grad
        elif second:
            l1 = norm(v1)
            l2 = norm(v2)
            lw = norm(w)
            v1xw = Math.cross(v1, w)
            v2xw = Math.cross(v2, w)
            cospv1 = dot(v1, w) / (l1 * lw)
            sinpv1 = np.sqrt(1 - cospv1 ** 2)
            cospv2 = -1 * dot(v2, w) / (l2 * lw)
            sinpv2 = np.sqrt(1 - cospv2 ** 2)
            mat = np.zeros((12, 12))
            if not (1e-6 < np.arccos(cospv1) < pi-1e-6) and \
                    (1e-6 < np.arccos(cospv2) < pi-1e-6):
                return phi, mat
            comp_coords = np.hstack([coords[self.i], coords[self.j], coords[self.k], coords[self.l]])
            for ii, i_coord in enumerate(comp_coords):
                for jj, j_coord in enumerate(comp_coords):
                    if ii // 3 == 3 and jj // 3 == 0:
                        mat[ii, jj] = 0
                        mat[jj, ii] = 0
                    elif jj <= ii:
                        # Is it the same atom?
                        ab = (ii // 3 == jj // 3)
                        i = ii % 3
                        j = jj % 3
                        k = (3 - (i + j)) - 1
                        t = list()
                        for s in [[ii, jj], [jj, ii]]:
                            v1j = v1[s[1] % 3]
                            v2j = v2[s[1] % 3]
                            wj = w[s[1] % 3]
                            t.append(zeta(i, 0, 1) * zeta(j, 0, 1) * v1xw[s[0] % 3] *
                                     (wj * cospv1 - v1j) / l1**2 / sinpv1**4)
                            t.append(zeta(i, 3, 2) * zeta(j, 3, 2) * v2xw[s[0] % 3] *
                                     (wj * cospv2 - v2j) / l2**2 / sinpv2**4)
                            t.append((zeta(i, 0, 1) * zeta(j, 3, 2) +
                                      zeta(i, 2, 3) * zeta(j, 1, 0)) *
                                     v1xw[i % 3] * (wj - 2 * v1j * cospv1 + wj * cospv1 ** 2) /
                                     (2 * l1 * lw * sinpv1 ** 4))
                            t.append((zeta(i, 3, 2) * zeta(j, 2, 1) +
                                      zeta(i, 2, 1) * zeta(j, 3, 2)) *
                                     v2xw[i % 3] * (wj + 2 * v1j * cospv2 + wj * cospv2 ** 2) /
                                     (2 * l2 * lw * sinpv2 ** 4))
                            t.append(zeta(i, 1, 2) * zeta(j, 2, 1) * v1xw[s[0] % 3] *
                                     (v1j + v1j * cospv1 ** 2 - 3 * wj * cospv1 + wj * cospv1 ** 3) /
                                     (2 * lw**2 * sinpv1**4))
                            t.append(zeta(i, 1, 2) * zeta(j, 2, 1) * v2xw[s[0] % 3] *
                                     (v2j + v2j * cospv2 ** 2 + 3 * wj * cospv2 + wj * cospv2 ** 3) /
                                     (2 * lw**2 * sinpv2**4))

                        t.append((1 - ab) * (zeta(i, 0, 1) * zeta(j, 1, 2) + zeta(i, 2, 1) * zeta(j, 1, 0)) *
                                 (j - i) * (-1/2) ** abs(j - i) * (w[k] * cospv1 - v1[k]) / (l1 * lw * sinpv1))
                        t.append((1 - ab) * (zeta(i, 3, 1) * zeta(j, 1, 2) + zeta(i, 2, 1) * zeta(j, 1, 0)) *
                                 (j - i) * (-1/2) ** abs(j - i) * (w[k] * cospv2 - v2[k]) / (l2 * lw * sinpv2))

                        val = sum(t)
                        mat[ii, jj] = val
                        mat[jj, ii] = val
            return phi, mat
        else:
            return phi

    def as_dict(self):
        dihedral_dict = {"@module": self.__class__.__module__,
                         "@class": self.__class__.__name__,
                         "i": self.i,
                         "j": self.j,
                         "k": self.k,
                         "l": self.l,
                         "idx": self.idx,
                         "weak": self.weak}
        return dihedral_dict

    @classmethod
    def from_dict(cls, d):
        dihedral = cls(d["i"], d["j"], d["k"], d["l"])
        if d["weak"] is not None:
            dihedral.weak = d["weak"]
        return dihedral


def get_clusters(C):
    nonassigned = list(range(len(C)))
    clusters = []
    while nonassigned:
        queue = {nonassigned[0]}
        clusters.append([])
        while queue:
            node = queue.pop()
            clusters[-1].append(node)
            nonassigned.remove(node)
            queue.update(n for n in np.flatnonzero(C[node]) if n in nonassigned)
    C = np.zeros_like(C)
    for cluster in clusters:
        for i in cluster:
            C[i, cluster] = True
    return clusters, C


class InternalCoords(object):
    def __init__(self, coords, fragments):
        self._coords = coords
        self.fragments = fragments

    @classmethod
    def from_geometry(cls, geom, dihedral=True, superweakdih=False):
        n = len(geom)
        geom = geom.supercell()
        dist = geom.dist(geom)
        radii = np.array([get_property(sp, 'covalent_radius') for sp in geom.species])
        bondmatrix = dist < 1.3 * (radii[None, :] + radii[:, None])
        fragments, C = get_clusters(bondmatrix)
        radii = np.array([get_property(sp, 'vdw_radius') for sp in geom.species])
        shift = 0.0
        C_total = C.copy()

        coords = list()
        bonds = list()
        while not C_total.all():
            bondmatrix |= ~C_total & (dist < radii[None, :]+radii[:, None]+shift)
            C_total = get_clusters(bondmatrix)[1]
            shift += 1.0
        for i, j in combinations(range(len(geom)), 2):
            if bondmatrix[i, j]:
                bond = Bond(i, j, C=C)
                bonds.append(bond)
                coords.append(bond)
        for j in range(len(geom)):
            for i, k in combinations(np.flatnonzero(bondmatrix[j, :]), 2):
                ang = Angle(i, j, k, C=C)
                if ang.eval(geom.coords) > pi/4:
                    coords.append(ang)
        if dihedral:
            for bond in bonds:
                cls.extend(coords,
                           get_dihedrals([bond.i, bond.j],
                                         geom.coords,
                                         bondmatrix,
                                         C,
                                         superweak=superweakdih))
        if geom.lattice is not None:
            coords, fragments = cls._reduce(n, coords, fragments)

        return cls(coords, fragments)

    def append(self, coord):
        self._coords.append(coord)

    @staticmethod
    def extend(coords, new_coords):
        return coords.extend(new_coords)

    def __iter__(self):
        return self._coords.__iter__()

    def __len__(self):
        return len(self._coords)

    @property
    def bonds(self):
        return [c for c in self if isinstance(c, Bond)]

    @property
    def angles(self):
        return [c for c in self if isinstance(c, Angle)]

    @property
    def dihedrals(self):
        return [c for c in self if isinstance(c, Dihedral)]

    @property
    def dict(self):
        return OrderedDict([
            ('bonds', self.bonds),
            ('angles', self.angles),
            ('dihedrals', self.dihedrals)
        ])

    def as_dict(self):
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "bonds": [b.as_dict() for b in self.bonds],
                "angles": [a.as_dict() for a in self.angles],
                "dihedrals": [d.as_dict() for d in self.dihedrals],
                "fragments": self.fragments}

    @classmethod
    def from_dict(cls, d):
        bonds = [Bond.from_dict(b) for b in d["bonds"]]
        angles = [Angle.from_dict(a) for a in d["angles"]]
        dihedrals = [Dihedral.from_dict(dd) for dd in d["dihedrals"]]
        fragments = d["fragments"]
        coords = bonds + angles + dihedrals
        return cls(coords, fragments)

    def __repr__(self):
        return "<InternalCoords '{}'>".format(', '.join(
            '{}: {}'.format(name, len(coords)) for name, coords in self.dict.items()
        ))

    def __str__(self):
        ncoords = sum(len(coords) for coords in self.dict.values())
        s = 'Internal coordinates:\n'
        s += '* Number of fragments: {}\n'.format(len(self.fragments))
        s += '* Number of internal coordinates: {}\n'.format(ncoords)
        for name, coords in self.dict.items():
            for degree, adjective in [(0, 'strong'), (1, 'weak'), (2, 'superweak')]:
                n = len([None for c in coords if min(2, c.weak) == degree])
                if n > 0:
                    s += '* Number of {} {}: {}\n'.format(adjective, name, n)
        return s.rstrip()

    def eval_geom(self, geom, template=None):
        geom = geom.supercell()
        q = np.array([coord.eval(geom.coords) for coord in self])
        if template is None:
            return q
        swapped = []  # dihedrals swapped by pi
        candidates = set()  # potentially swapped angles
        for i, dih in enumerate(self):
            if not isinstance(dih, Dihedral):
                continue
            diff = q[i] - template[i]
            if abs(abs(diff) - 2 * pi) < pi / 2:
                q[i] -= 2 * pi * np.sign(diff)
            elif abs(abs(diff) - pi) < pi/2:
                q[i] -= pi*np.sign(diff)
                swapped.append(dih)
                candidates.update(dih.angles)
        for i, ang in enumerate(self):
            if not isinstance(ang, Angle) or ang not in candidates:
                continue
            # candidate angle was swapped if each dihedral that contains it was
            # either swapped or all its angles are candidates
            if all(dih in swapped or all(a in candidates for a in dih.angles)
                   for dih in self.dihedrals if ang in dih.angles):
                q[i] = 2 * pi - q[i]
        return q

    @staticmethod
    def _reduce(n, c, f):
        idxs = np.int64(np.floor(np.array(range(3 ** 3 * n)) / n))
        idxs, i = np.divmod(idxs, 3)
        idxs, j = np.divmod(idxs, 3)
        k = idxs % 3
        ijk = np.vstack((i, j, k)).T-1
        coords = [
            coord for coord in c
            if np.all(np.isin(coord.center(ijk), [0, -1]))
        ]
        idxs = set(i for coord in coords for i in coord.idx)
        fragments = [frag for frag in f if set(frag) & idxs]
        return coords, fragments

    def hessian_guess(self, geom):
        geom = geom.supercell()
        rho = geom.rho()
        return np.diag([coord.hessian(rho) for coord in self])

    def weights(self, geom):
        geom = geom.supercell()
        rho = geom.rho()
        return np.array([coord.weight(rho, geom.coords) for coord in self])

    def B_matrix(self, geom):
        geom = geom.supercell()
        B = np.zeros((len(self), len(geom), 3))
        for i, coord in enumerate(self):
            _, grads = coord.eval(geom.coords, grad=True)
            idx = [k % len(geom) for k in coord.idx]
            for j, grad in zip(idx, grads):
                B[i, j] += grad
        return B.reshape(len(self), 3 * len(geom))

    def K_matrix(self, geom, grad):
        geom = geom.supercell()
        K = np.zeros((len(geom) * 3, len(geom) * 3))

        for ii, coord in enumerate(self):
            _, mat = coord.eval(geom.coords, second=True)
            idx = [k % len(geom) for k in coord.idx]
            for kk, row in enumerate(mat):
                for ll, val in enumerate(row):
                    krow = idx[kk // 3] * 3 + kk % 3
                    kcol = idx[ll // 3] * 3 + ll % 3
                    K[krow, kcol] += val * grad[ii]

        return K

    def update_geom(self, geom, q, dq, B_inv, log=lambda _: None):
        geom = geom.copy()
        thre = 1e-6
        # target = CartIter(q=q+dq)
        # prev = CartIter(geom.coords, q, dq)
        for i in range(100):
            coords_new = geom.coords + B_inv.dot(dq).reshape(-1, 3) / angstrom
            dcart_rms = Math.rms(coords_new - geom.coords)
            geom.coords = coords_new
            q_new = self.eval_geom(geom, template=q)
            dq_rms = Math.rms(q_new - q)
            q, dq = q_new, dq - (q_new - q)
            if dcart_rms < thre:
                msg = 'Perfect transformation to cartesians in {} iterations'
                break
            if i == 0:
                keep_first = geom.copy(), q, dcart_rms, dq_rms
        else:
            # Should there be an exception here?
            msg = 'Transformation did not converge in {} iterations'
            geom, q, dcart_rms, dq_rms = keep_first
        log(msg.format(i+1))
        log('* RMS(dcart): {:.3}, RMS(dq): {:.3}'.format(dcart_rms, dq_rms))
        return q, geom


def get_dihedrals(center, coords, bondmatrix, C, superweak=False):
    neigh_l = [n for n in np.flatnonzero(bondmatrix[center[0], :]) if n not in center]
    neigh_r = [n for n in np.flatnonzero(bondmatrix[center[-1], :]) if n not in center]
    angles_l = [Angle(i, center[0], center[1]).eval(coords) for i in neigh_l]
    angles_r = [Angle(center[-2], center[-1], j).eval(coords) for j in neigh_r]
    nonlinear_l = [n for n, ang in zip(neigh_l, angles_l) if ang < pi-1e-3 and ang >= 1e-3]
    nonlinear_r = [n for n, ang in zip(neigh_r, angles_r) if ang < pi-1e-3 and ang >= 1e-3]
    linear_l = [n for n, ang in zip(neigh_l, angles_l) if ang >= pi-1e-3 or ang < 1e-3]
    linear_r = [n for n, ang in zip(neigh_r, angles_r) if ang >= pi-1e-3 or ang < 1e-3]
    assert len(linear_l) <= 1
    assert len(linear_r) <= 1
    if center[0] < center[-1]:
        nweak = len(list(
            None for i in range(len(center)-1)
            if not C[center[i], center[i+1]]
        ))
        dihedrals = []
        for nl, nr in product(nonlinear_l, nonlinear_r):
            if nl == nr:
                continue
            weak = nweak + \
                (0 if C[nl, center[0]] else 1) + \
                (0 if C[center[0], nr] else 1)
            if not superweak and weak > 1:
                continue
            dihedrals.append(Dihedral(
                nl,
                center[0],
                center[-1],
                nr,
                weak=weak,
                angles=(
                    Angle(nl, center[0], center[1], C=C),
                    Angle(nl, center[-2], center[-1], C=C)
                )
            ))
    else:
        dihedrals = []
    if len(center) > 3:
        pass
    elif linear_l and not linear_r:
        dihedrals.extend(get_dihedrals(linear_l + center, coords, bondmatrix, C))
    elif linear_r and not linear_l:
        dihedrals.extend(get_dihedrals(center + linear_r, coords, bondmatrix, C))
    return dihedrals


def zeta(a, b, c):
    return int(a == b) - int(a == c)