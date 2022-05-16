from PyPreludium.utils import get_beta
from PyPreludium.preluts import NeutralPreLUT
import numpy as np
import struct
from PyPreludium.tests.test_files import tfp
from numpy import newaxis as na
import numpy.testing as npt


def load_prelut_file(f):
    with open(f, 'rb') as fid:
        fid.seek(-1, 2)     # go to the file end.
        eof = fid.tell()   # get the end of file location
        fid.seek(0, 0)      # go back to file beginning

        def read_complex(shape):
            n = np.prod(shape)
            v = np.reshape(struct.unpack('d' * 2 * n, fid.read(16 * n)), shape + (2,))
            return np.sum(v * np.array([1, 1j]), -1)

        def read_level():
            r = ([read_complex((6, 6)) for _ in range(3)] +   # Yleft, Rleft, Rright
                 [read_complex((6,)) for _ in range(6)] +   # dyxu0, dyxu1, dyxv0, dyxv1, dyxw0, dyxw1
                 list(struct.unpack('ddi', fid.read(20))))  # sleft, sright, level
            struct.unpack('i', fid.read(4))
            return r
        r = []
        while fid.tell() < eof:
            if fid.tell() == 467124:
                print(fid.tell())
            r.append(read_level())

    class Prelut(dict):
        def __getattr__(self, k):
            return self[k]

    return Prelut({k: np.array(v) for k, v in zip(['Yleft', 'Rleft', 'Rright',
                                                            'dyxu0', 'dyxu1', 'dyxv0', 'dyxv1', 'dyxw0', 'dyxw1',
                                                            'sleft', 'sright', 'level'],
                                                  zip(*r))})


def test_load_prelut_file():
    prelut = load_prelut_file(tfp + '0.0000-09.0000.pre')

    npt.assert_array_almost_equal(prelut.Yleft[7][0, 3], -0.330350424728106 + 9.114133344026951e-012j, 10)
    npt.assert_array_almost_equal(prelut.Rleft[7][1, 4], -0.139000795854240 - 1.272631906082874E-010j, 10)
    npt.assert_array_almost_equal(prelut.Rright[7][0, 3], 3.768307078547423e-002 + 1.581796234292119e-010j, 10)
    npt.assert_array_almost_equal(prelut.dyxu0[7][1], -2.423313312692920e-011 - 1.267852505176752e-021j, 10)
    npt.assert_array_almost_equal(prelut.dyxu1[7][1], -3.528960438325056e-020 - 1.849660676125115e-030j, 10)
    npt.assert_array_almost_equal(prelut.dyxv0[7][3], 6.806655201915466e-011 - 7.883109477872014e-021, 10)
    npt.assert_array_almost_equal(prelut.dyxv1[7][3], 9.905991452339835e-020 - 1.147145744025741e-029, 10)
    npt.assert_array_almost_equal(prelut.dyxw0[7][2], 1.999045719443342e-030 - 2.122488476172560e-021, 10)
    npt.assert_array_almost_equal(prelut.dyxw1[7][2], 2.912345039036901e-039 - 3.093036809788905e-030, 10)
    npt.assert_array_almost_equal(prelut.dyxw0[7][2], 1.999045719443342e-030 - 2.122488476172560e-021, 10)
    npt.assert_array_almost_equal(prelut.sleft[7], 0.35, 10)
    npt.assert_array_almost_equal(prelut.sright[7], 0.4, 10)
    npt.assert_array_almost_equal(prelut.level[7], 7, 10)



# A(0)
B = np.array([complex(0.997445506990155, 0.000000000000000e+000),
              complex(-4.987227534950781e-002, 1.690728946293729e-014),
              complex(1.465703679329536e-039, -2.124682676392099e-052),
              complex(-7.328518396647689e-041, 3.546742628863952e-053),
              complex(7.792543023360604e-003, -1.292252580963615e-012),
              complex(1.280398450137159e-024, 3.222891189714300e-012)])
# yleft(:,6)
C = np.array([complex(1.047546898171445e-022, -4.101687702293994e-011),
              complex(-9.846483766668896e-025, -1.949161450360140e-011),
              complex(3.970641630432963e-061, -6.027069775156658e-050),
              complex(-1.360387977384101e-062, -2.864026273596058e-050),
              complex(2.052590442281348e-020, -1.583918971177244e-013),
              complex(1.00000000000000, -1.069408260473665e-035)])

# A = (Yleft[:, j] * np.conj(Yleft[:, j])[:, na])
# >>> np.dot(A[0], Yleft[:, 5])
# (2.656407250768623e-22-4.316413399715623e-11j)

#Yleft[:, j + 1:] = Yleft[:, j + 1:] - np.dot((Yleft[:, j] * np.conj(Yleft[:, j])[:, na]), Yleft[:, j + 1:])
# >>> Yleft[0,5]
# (-1.6088603525971778e-22+2.1472569742162893e-12j)
# (-1.611357713210343D-022,-4.298525405212307D-012)


# next.dat.Rleft[j, j + 1:] = np.conj(Yleft[:, j])@Yleft[:, j + 1:]
# >>> next.dat.Rleft[4,5]
# (-2.8855565613824884e-23-1.500305359989711e-10j)
#np.sum(np.conj(Yleft[(0,1,5), 4]) * Yleft[(0,1,5), 5])
#(-2.8920903963399594e-23-1.500305359989711e-10j)

# >>> Yleft[(0,1,5), 4]
# array([ 4.99376169e-02+8.61709898e-13j,  9.98752339e-01+9.04871026e-13j,
#        -1.48646439e-23+1.28514955e-10j])
# (4.993761694389232D-002,7.324665664985942D-017)
# (0.998752338877845,1.383487270702489D-013)
# (-1.315350765671031D-023,1.281765071710166D-010)
# >>> Yleft[(0,1,5), 5]
# array([-2.19825973e-24+2.14825814e-12j,  3.79741961e-24-2.16498713e-11j,
#         1.00000000e+00-1.29807574e-33j])
# (2.339688079440132D-025,3.177741907478460D-012)
# (-4.597372173620277D-024,1.281777406966873D-010)
# (1.00000000000000,-1.459739419215753D-034)


# B = np.triu(next.dat.Rleft / np.diag(next.dat.Rleft), 1)
#>>> next.dat.Rleft[4,5]
#(-2.8855565613824884e-23-1.500305359989711e-10j)
# (-1.140090232913836D-023,-1.496920878380001D-010)

#self.dat.Rright[i, j] -= np.sum(self.dat.Rright[i, i:j] * B[i:j, j])
# >>> B[4,5]
# (-2.8855565613824884e-23-1.500305359989711e-10j)
# (-1.140090232913836D-023,-1.496920878380001D-010)

# 0 Rright 3.3887095708951127e-13 2.9769347466917524 (4, 5)
#(2.889161255556684e-23-1.5021795710389924e-10j)
#(1.1415144561211173e-23-1.4987908614680973e-10j)

def test_prelut():
    res = NeutralPreLUT(zeta0=0, kz0=1e-9, beta=get_beta(np.array([0]))[0],
                        kzmax=300, accgoal=0.0001, ds=0.05).make_prelut()
    prelut = load_prelut_file(tfp + '0.0000-09.0000.pre')
    for l in range(369):
        for v in res:
            A, B = np.atleast_2d(res[v][l].values), np.atleast_2d(prelut[v][l])
            err = np.abs(A - B)
            rerr = np.abs(err / np.mean([A, B], 0))
            i, j = np.unravel_index(np.argmax(err), A.shape)
            print(l, v, err.max(), np.nanmax(rerr), (i, j))
            if np.nanmax(rerr) > .1:
                print(A[i, j])
                print(B[i, j])
            #npt.assert_array_almost_equal(res[v][l], prelut[v][l])
        # for l, l in enumerate(res[v].level):
        #     print(l, np.abs(res[v][l] - prelut[v][l]).max().item())
        #     #npt.assert_array_almost_equal(res[v][l], prelut[v][l], 4, err_msg=f'{v}, {l}')

    print()
