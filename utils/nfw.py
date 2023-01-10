"""Functions for the NFW halo."""

import jax.numpy as jnp
from jax import jit, vmap

import sys
sys.path.append('..')
from utils.units_constants import *
from utils.geometry import Gr


####################
## constants

rs_NFW = 16 # [kpc]
rho_r_Sun = 0.46 # [GeV/cm^3]
norm_NFW = 1 / ( (r_Sun/rs_NFW) * (1+r_Sun/rs_NFW)**2 )
intg_d_s = jnp.concatenate((jnp.linspace(0, 20, 200, endpoint=False),
                            jnp.linspace(20, 200, 200, endpoint=False),
                            jnp.linspace(200, 1000, 100, endpoint=True)))

####################
## functions

@jit
def rho_NFW(r):
    """NFW halo density rho [GeV/cm^3] as a function of distance to galactic
    center r [kpc]. Vectorized manually.
    """
    return rho_r_Sun / ( (r/rs_NFW)*(1+r/rs_NFW)**2 ) / norm_NFW

rho_integral_ref = rho_NFW(r_Sun) * 10 # [GeV/cm^3] * [kpc] = [GeV kpc/cm^3]

@jit
@vmap
def rho_integral(lb):
    """NFW halo density integral [GeV kpc/cm^3] as a function of galactic
    coordinates (l, b) [rad, rad]. Vmapped.
    """
    l, b = lb # single numbers
    lbd_s = jnp.stack([ jnp.full_like(intg_d_s, l),
                        jnp.full_like(intg_d_s, b),
                        intg_d_s ], axis=-1)
    rho_s = rho_NFW(Gr(lbd_s))
    return jnp.trapz(rho_s, intg_d_s)