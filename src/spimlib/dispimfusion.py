"""reimplementation of the diSPIMFusion API, as first described in [1]

References
---
[1] Guo M., et al. "Rapid image deconvolution and multiview fusion..."
    Nature Biotechnology 38.11 (2020): 1337-1346.
[2] https://github.com/eguomin/diSPIMFusion
[3] https://github.com/eguomin/microImageLib
"""
## IMPORTS
## import order is roughly that of the pipeline that is used by
## the diSPIMFusion plugin

## deskewing
## ===
## deskew the inputs into the "normal" LSFM coordinate system
##   deskewing tends to produce a lot of black space in the output
##   so then crop down to only the parts of the output with actual data
from .deskew.dispim import deskew_stage_scan  # deskewing
from . import roi

## isotropization
## ===
## deskewed output has voxel size [step_size, pixel_size, pixel_size]
## and before deconvolution this has to be isotropic.
##   can do this by either real-space interpolating (`upsample_interp`)
##   or by padding the FFT and re-transforming (`upsample_fourier`)
##   the diSPIMFusion plugin ([2]) uses real-space interpolation
from .isotropize import interpolate as upsample_interp
from .isotropize import fourier_upsample as upsample_fourier

## rotation
## ===
## for 2 orthogonal objectives, one must be rotated into the coordinate
## system of the other. typically, this is done by rotating the 'B'
## head into the frame of 'A' by rotating 90 degrees
from ._rotate_y import rotate_view  # view B rotation

## registration
## ===
## the 2 views must be registered to one another prior to deconvolution
##   to get a good registration, use powell's method to optimize an objective
##   function, such as the correlation ratio.
##   preliminary alignments can be obtained by using phase cross correlation
from ._util import pad_to_same_size
from . import reg
from .interp.affine import transform as affine_transform

## deconvolution
## ===
## to achieve isotropic resolution, the two registered views must be
## co-deconvolved. because diSPIMFusion uses the `joint_rl_dispim`
## deconvolution algorithm, we alias it as `deconvolve` here
from ._util import shared_bbox_from_proj_threshold
from .decon.util import crop_and_pad_for_deconv
from .decon.rl.dualview_fft import joint_rl_dispim as deconvolve 


## rotation 2
## ===
## *optional*: the final output can be rotated so that it is in the "normal"
##   lab frame coordinate system where xy is the coverslip and z is normal
##   to the coverslip
# TODO: implement rotation with newer version of code