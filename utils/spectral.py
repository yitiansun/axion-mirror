import jax.numpy as jnp
from jax import jit, vmap

import sys
sys.path.append('..')
from utils.units_constants import *

##############################
## spectral analysis
fDelta = 0.721 # frequency domain cut associated with the above bandwidth (2.17)
def dnu(nu):
    """[MHz]([MHz])"""
    return 2.17 * sigmad_over_c * nu