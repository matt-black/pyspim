import warnings

import cupy
import numpy
from pywt import dwt2, idwt2
from tqdm.auto import trange

from ..typing import NDArray


def remove_stripes(
    im: NDArray,
    dec_num: int,
    wavelet: str,
    sigma: float,
    direction: str = "horizontal",
    verbose: bool = False,
) -> NDArray:
    """remove stripes from input image by the FFTWavelet algorithm

    :param im: input image
    :type im: NDArray
    :param dec_num: number of decimation levels (loops)
    :type dec_num: int
    :param wavelet: wavelet to use
    :type wavelet: str
    :param sigma: _description_
    :type sigma: float
    :param direction: direction of stripes in input to be removed, defaults to 'horizontal'
    :type direction: str, optional
    :param verbose: show progress through loop, defaults to False
    :type verbose: bool, optional
    :raises ValueError: if direction isn't `'horizontal'` or `'vertical'`
    :return: input image with stripes removed
    :rtype: NDArray
    """
    input_is_gpu = cupy.get_array_module(im) == cupy
    if input_is_gpu:
        warnings.warn("remove_stripes is computed on the cpu, will move gpu->cpu->gpu")
    im = im.get() if input_is_gpu else im
    dir_char = direction.lower()[0]
    scales = []
    prev_cA = im.copy()
    for _ in trange(dec_num, desc="dwt") if verbose else range(dec_num):
        (cA, (cH, cV, cD)) = dwt2(prev_cA, wavelet)
        if dir_char == "v" or dir_char == "c":
            fCv = numpy.fft.fftshift(numpy.fft.fft2(cV))
            my, mx = fCv.shape
            flrmy2 = numpy.floor(my / 2)
            denom = 2 * numpy.square(sigma)
            damp = 1 - numpy.exp(
                -numpy.square(numpy.arange(-flrmy2, -flrmy2 + my)) / denom
            )
            fCv *= numpy.repeat(damp[:, None], mx, 1)
            cV = numpy.fft.ifft(numpy.fft.ifftshift(fCv))
        elif dir_char == "h" or dir_char == "r":
            fCh = numpy.fft.fftshift(numpy.fft.fft2(cH))
            my, mx = fCh.shape
            flrmx2 = numpy.floor(mx / 2)
            denom = 2 * numpy.square(sigma)
            damp = 1 - numpy.exp(
                -numpy.square(numpy.arange(-flrmx2, -flrmx2 + mx)) / denom
            )
            fCh *= numpy.repeat(damp[None, :], my, 0)
            cH = numpy.fft.ifft(numpy.fft.ifftshift(fCh))
        else:
            raise ValueError("invalid direction")
        prev_cA = cA.copy()
        scales.insert(0, (cH, cV, cD))
    recon = prev_cA
    for dec in trange(dec_num, desc="idwt") if verbose else range(dec_num):
        cH, cV, cD = scales[dec]
        recon = recon[: cH.shape[0], : cH.shape[1]]
        recon = idwt2((recon, (cH, cV, cD)), wavelet)
    recon = numpy.real(recon)
    if input_is_gpu:
        return cupy.asarray(recon)
    return recon
