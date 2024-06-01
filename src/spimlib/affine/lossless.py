"""
Implementation of "Practically Lossless Affine Image Transformation"

References
---
[1] D. Pflugfelder and H. Scharr, "Practically Lossless Affine Image Transformation," in IEEE Transactions on Image Processing, vol. 29, pp. 5367-5373, 2020, doi: 10.1109/TIP.2020.2982260.
"""
import cupy
import numpy


def scale_factor(img, pad_size, upsample=True):
    s = numpy.asarray(img.shape, float)
    if upsample:
        return numpy.prod(s+pad_size) / numpy.prod(s)
    else:
        return numpy.prod(s-pad_size) / numpy.prod(s)
