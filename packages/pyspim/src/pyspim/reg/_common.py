from collections.abc import Callable

import numpy

from .._matrix import (
    rotation_about_point_matrix,
    rotation_matrix,
    scale_matrix,
    shear_about_point_matrix,
    symmetric_shear_about_point_matrix,
    translation_matrix,
)

def _trans_matrix(params, x, y, z):
    return translation_matrix(*params)


def _trans_scale_matrix(params, x, y, z):
    T = translation_matrix(*params[:3])
    T[0, 0], T[1, 1], T[2, 2] = params[3], params[4], params[5]
    return T


def _rot_trans_matrix(params, x, y, z):
    R = rotation_about_point_matrix(*(params[3:] * numpy.pi / 180), 
                                    x, y, z) # type: ignore
    T = translation_matrix(*params[:3])
    return T @ R


def _rot0_trans_matrix(params, x, y, z):
    R = rotation_matrix(*(params[3:] * numpy.pi / 180))
    T = translation_matrix(*params[:3])
    return T @ R


def _rot_trans_scale_matrix(params, x, y, z):
    R = rotation_about_point_matrix(*(params[3:6] * numpy.pi / 180), 
                                    x, y, z) # type: ignore
    S = scale_matrix(*params[6:])
    T = translation_matrix(*params[:3])
    return T @ R @ S


def _rot0_trans_scale_matrix(params, x, y, z):
    R = rotation_matrix(*(params[3:6] * numpy.pi / 180))
    S = scale_matrix(*params[6:])
    T = translation_matrix(*params[:3])
    return T @ R @ S


def _symmshear_trans_matrix(params, x, y, z):
    S = symmetric_shear_about_point_matrix(
        *(numpy.tan(params[3:] * numpy.pi / 180)), x, y, z # type: ignore
    )
    T = translation_matrix(*params[:3])
    return T @ S


def _shear_trans_matrix(params, x, y, z):
    S = shear_about_point_matrix(*(numpy.tan(params[3:] * numpy.pi / 180)), 
                                 x, y, z) # type: ignore
    T = translation_matrix(*params[:3])
    return T @ S


def _shear_trans_scale_matrix(params, x, y, z):
    Sh = shear_about_point_matrix(*(numpy.tan(params[3:9] * numpy.pi / 180)), 
                                  x, y, z) # type: ignore
    T = translation_matrix(*params[:3])
    S = scale_matrix(*params[9:])
    return T @ S @ Sh


def parse_transform_string(
    string: str
) -> tuple[Callable, Callable, list[float]]:
    if string in ("t", "trans", "translation"):
        postfix = lambda x: {"t": [f"{v:.2f}" for v in x[:3]]}
        return (
            _trans_matrix,
            postfix,
            [
                0,
            ]
            * 3,
        )
    elif string in ("t+s", "transscale", "translation+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "s": [f"{v:.2f}" for v in x[3:]],
        }
        return _trans_scale_matrix, postfix, [0, 0, 0, 1, 1, 1]
    # rotation + translation (+ scaling)
    elif string in ("t+r", "transrot", "translation+rotation"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:]],
        }
        par0 = [
            0,
        ] * 6
        return (
            _rot_trans_matrix,
            postfix,
            [
                0,
            ]
            * 6,
        )
    elif string in ("t+r0", "transrot0", "translation+rotation0"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:]],
        }
        par0 = [
            0,
        ] * 6
        return (
            _rot0_trans_matrix,
            postfix,
            [
                0,
            ]
            * 6,
        )
    elif string in ("t+r+s", "transrotscale", "translation+rotation+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:6]],
            "s": [f"{v:.2f}" for v in x[6:]],
        }
        par0 = [
            0.,
        ] * 6 + [
            1.,
        ] * 3
        return _rot_trans_scale_matrix, postfix, par0
    elif string in ("t+r0+s", "transrot0scale", "translation+rotation0+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "a": [f"{v:.2f}" for v in x[3:6]],
            "s": [f"{v:.2f}" for v in x[6:]],
        }
        par0 = [
            0.,
        ] * 6 + [
            1.,
        ] * 3
        return _rot0_trans_scale_matrix, postfix, par0
    # shear + translation
    elif string in ("t+ssh", "transsymmshear", "translation+symmetricshear"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "sh": [f"{v:.2f}" for v in x[3:]],
        }
        return (
            _symmshear_trans_matrix,
            postfix,
            [
                0,
            ]
            * 6,
        )
    elif string in ("t+sh", "transshear", "translation+shear"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "sh": [f"{v:.2f}" for v in x[3:]],
        }
        return (
            _shear_trans_matrix,
            postfix,
            [
                0,
            ]
            * 9,
        )
    elif string in ("t+sh+s", "transshearscale", "translation+shear+scale"):
        postfix = lambda x: {
            "t": [f"{v:.2f}" for v in x[:3]],
            "sh": [f"{v:.2f}" for v in x[3:9]],
            "s": [f"{v:.2f}" for v in x[9:]],
        }
        par0 = [
            0.,
        ] * 9 + [
            1.,
        ] * 3
        return _shear_trans_scale_matrix, postfix, par0
    else:
        raise ValueError("invalid transform string")
