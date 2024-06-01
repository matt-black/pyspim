import math
import numpy


## rotation matrices
def _yaw_matrix(alpha):
    return numpy.vstack([
        numpy.array([math.cos(alpha), -math.sin(alpha), 0]),
        numpy.array([math.sin(alpha),  math.cos(alpha), 0]),
        numpy.array([0, 0, 1])
    ])


def _pitch_matrix(beta):
    return numpy.vstack([
        numpy.array([math.cos(beta), 0, math.sin(beta)]),
        numpy.array([0, 1, 0]),
        numpy.array([-math.sin(beta), 0, math.cos(beta)])
    ])


def _roll_matrix(gamma):
    return numpy.vstack([
        numpy.array([1, 0, 0]),
        numpy.array([0, math.cos(gamma), -math.sin(gamma)]),
        numpy.array([0, math.sin(gamma), math.cos(gamma)])
    ])


def rotation_matrix(alpha, beta, gamma):
    yaw = _yaw_matrix(alpha)
    pitch = _pitch_matrix(beta)
    roll = _roll_matrix(gamma)
    R = yaw @ pitch @ roll
    out = numpy.eye(4)
    out[:3,:3] = R
    return out


def rotation_about_point_matrix(alpha, beta, gamma, x, y, z):
    t1 = translation_matrix(x, y, z)
    R = rotation_matrix(alpha, beta, gamma)
    t2 = translation_matrix(-x, -y, -z)
    return t1 @ R @ t2


## translation matrix
def translation_matrix(x, y, z):
    T = numpy.eye(4)
    T[0,3], T[1,3], T[2,3] = x, y, z
    return T


def rigid_affine_matrix(alpha, beta, gamma, x, y, z):
    R = rotation_matrix(alpha, beta, gamma)
    R[:,-1] = numpy.array([x, y, z, 1])
    return R


## scale matrix
def scale_matrix(sx, sy, sz):
    return numpy.diag([sx, sy, sz, 1.])
    

def rotation_translation_scale_matrix(alpha, beta, gamma,
                                      x, y, z, sx, sy, sz):
    Rt = rotation_translation_matrix(alpha, beta, gamma, x, y, z)
    S = scale_matrix(sx, sy, sz)
    return S @ Rt


## shear matrices
def _shear_x_matrix(sy, sz):
    return numpy.vstack([
        numpy.array([1, math.tan(sy), math.tan(sz), 0]),
        numpy.array([0, 1, 0, 0]),
        numpy.array([0, 0, 1, 0]),
        numpy.array([0, 0, 0, 1])
    ])


def _shear_y_matrix(sx, sz):
    return numpy.vstack([
        numpy.array([1, 0, 0, 0]),
        numpy.array([math.tan(sx), 1, math.tan(sz), 0]),
        numpy.array([0, 0, 1, 0]),
        numpy.array([0, 0, 0, 1])
    ])

def _shear_z_matrix(sx, sy):
    return numpy.vstack([
        numpy.array([1, 0, 0, 0]),
        numpy.array([0, 1, 0, 0]),
        numpy.array([math.tan(sx), math.tan(sy), 1, 0]),
        numpy.array([0, 0, 0, 1])
    ])


def shear_matrix(sxy, sy, sz):
    X = _shear_x_matrix(sy, sz)
    Y = _shear_y_matrix(sx, sz)
    Z = _shear_z_matrix(sx, sy)
    return X @ Y @ Z
