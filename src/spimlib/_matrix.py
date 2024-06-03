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
def _shear_x_matrix(h_xy, h_xz):
    return numpy.vstack([
        numpy.array([1, h_xy, h_xz, 0]),
        numpy.array([0, 1, 0, 0]),
        numpy.array([0, 0, 1, 0]),
        numpy.array([0, 0, 0, 1])
    ])


def _shear_y_matrix(h_yx, h_yz):
    return numpy.vstack([
        numpy.array([1, 0, 0, 0]),
        numpy.array([h_yx, 1, h_yz, 0]),
        numpy.array([0, 0, 1, 0]),
        numpy.array([0, 0, 0, 1])
    ])


def _shear_z_matrix(h_zx, h_zy):
    return numpy.vstack([
        numpy.array([1, 0, 0, 0]),
        numpy.array([0, 1, 0, 0]),
        numpy.array([h_zx, h_zy, 1, 0]),
        numpy.array([0, 0, 0, 1])
    ])


def shear_matrix(h_xy, h_xz, h_yx, h_yz, h_zx, h_zy):
    X = _shear_x_matrix(h_xy, h_xz)
    Y = _shear_y_matrix(h_yx, h_yz)
    Z = _shear_z_matrix(h_zx, h_zy)
    return X @ Y @ Z


def symmetric_shear_matrix(h_xyx, h_xzx, h_yzy):
    return shear_matrix(h_xyx, h_xzx, h_yxy, h_yzy, h_xzx, h_yzy)


def shear_about_point_matrix(h_xy, h_xz, h_yx, h_yz, h_zx, h_zy,
                             x, y, z):
    t1 = translation_matrix(x, y, z)
    S = shear_matrix(h_xy, h_xz, h_yx, h_yz, h_zx, h_zy)
    t2 = translation_matrix(-x, -y, -z)
    return t1 @ S @ t2


def symmetric_shear_about_point_matrix(h_xyx, h_xzx, h_yzy, x, y, z):
    t1 = translation_matrix(x, y, z)
    S = symmetric_shear_matrix(h_xyx, h_xzx, h_yzy)
    t2 = translation_matrix(-x, -y, -z)
    return t1 @ S @ t2
