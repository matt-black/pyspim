from typing import Optional

from ..typing import NDArray
from .._util import get_skimage_module


def richardson_lucy(image : NDArray, psf : NDArray,
                    num_iter : int=50, clip : bool=True,
                    filter_epsilon : Optional[float]=None) -> NDArray:
    """Richardson-Lucy deconvolution
        this is just a thin wrapper around cucim/skimage that dispatches
        to the correct library based on the input being already on the
        cpu (`skimage`) or gpu (`cucim`)

    :param image: input image/volume
    :type image: NDArray
    :param psf: point spread function
    :type psf: NDArray
    :param num_iter: number of iterations
    :type num_iter: int
    :param clip: if `True`, pixel values >1 or <-1 are thresholded
        (for skimage pipeline compatability)
    :type clip: bool
    :param filter_epsilon: values below which intermediate results become 0
        (avoids division by small numbers errors)
    :type filter_epsilon: Optional[float]
    :returns: deconvolved image
    :rtype: NDArray
    """
    skimage = get_skimage_module(image)
    return skimage.restoration.richardson_lucy(
        image, psf, num_iter, clip, filter_epsilon
    )
