from berny.geomlib import Geometry
from berny.coords import InternalCoords, angstrom


def test_cycle_dihedrals():
    geom = Geometry.from_atoms([(ws[1], ws[2:5]) for ws in (l.split() for l in """\
    1 H -0.000000000000 0.000000000000 -1.142569988888
    2 O 1.784105551801 1.364934064507 -1.021376180623
    3 H 2.248320553963 2.318104360291 -2.500037742933
    4 H 3.285761299420 0.674554743661 -0.259576564237
    5 O -1.784105551799 -1.364934064536 -1.021376180591
    6 H -2.248320553963 -2.318104360291 -2.500037742933
    7 H -3.285761299424 -0.674554743614 -0.259576564287
    8 O 5.839754502206 -0.500682935209 1.037064691223
    9 H 7.440059622286 -1.597667062287 0.565115038647
    10 H 6.475526400773 0.638572472561 2.500357106648
    11 O -5.839754502205 0.500682935191 1.037064691242
    12 H -7.440059622286 1.597667062287 0.565115038647
    13 H -6.475526400773 -0.638572472561 2.500357106648
    """.strip().split('\n'))], unit=1/angstrom)
    coords = InternalCoords.from_geometry(geom)
    assert not [dih for dih in coords.dihedrals if len(set(dih.idx)) < 4]
