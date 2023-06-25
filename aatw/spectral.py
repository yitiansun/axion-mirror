"""Functions for spectrum"""

import sys
sys.path.append("..")

from aatw.units_constants import hbar, c0, sigmad_over_c, gagg_CAST


fDelta = 0.721 # frequency domain cut associated with the above bandwidth (2.17)

def dnu(nu):
    """[MHz]([MHz]). Input can be a vector."""
    return 2.17 * sigmad_over_c * nu


def prefac(nu):
    """Prefactor = (fDelta/dnu(nu)) * hbar * c0**4 * gagg_CAST**2 / 16.
    Units: [MHz^-1] [cm^2 MHz g] [cm MHz]^4 [cm^-2 MHz^-2 g^-1]^2
         = [g^-1 cm^2].
    Note that since integral of rho ~ [g cm^-2], the product is ~ [1].
    Input can be a vector.
    """
    return (fDelta/dnu(nu)) * hbar * c0**4 * gagg_CAST**2 / 16