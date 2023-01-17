import jax.numpy as jnp

import sys
sys.path.append('..')
from utils.units_constants import *

## units
# length: kpc
# B field: muGauss

####################
## B field

def B_disk_ASR(stz_s):
    """B_disk in Axi-symmetric spiral model + ring model (ASR) from NE2001.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    R_s, t_s, z_s = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    R0, z0, Rc, Rsun = 10, 1, 5, 8.5 # all [kpc]
    B0, Bc = 2, 2 # all [muG]
    
    D1_s = jnp.where(R_s < Rc, Bc, B0*jnp.exp( -(R_s-Rsun)/R0-jnp.abs(z_s)/z0 ))
    D2_s = jnp.array(
        jnp.logical_or( (7.5<R_s), jnp.logical_and(5<R_s, R_s<6) ),
        dtype=float) * 2 - 1
    BDt_s = - D1_s * D2_s
    return jnp.stack([jnp.zeros_like(BDt_s),
                      BDt_s,
                      jnp.zeros_like(BDt_s)], axis=-1)

def B_disk_BS(stz_s):
    """B_disk in Bi-symmetric spiral model (BS) from NE2001.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    R_s, t_s, z_s = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    R0, z0, Rc, Rsun = 6, 1, 3, 8.5 # all [kpc]
    B0, Bc = 2, 2 # all [muG]
    
    D1_s = jnp.where(R_s < Rc, Bc, B0*jnp.exp( -(R_s-Rsun)/R0-jnp.abs(z_s)/z0 ))
    Rb_s = jnp.where(R_s > 6,  9,  6)
    p_s  = jnp.where(R_s > 6, jnp.deg2rad(-10), jnp.deg2rad(-15))
    beta_s = 1/jnp.tan(p_s)
    D2_s = -jnp.cos(t_s + beta_s*jnp.log(R_s/Rb_s))
    D1D2_s = D1_s*D2_s
    return jnp.stack([D1D2_s*jnp.sin(p_s),
                      -D1D2_s*jnp.cos(p_s),
                      jnp.zeros_like(D1D2_s)], axis=-1)

def B_halo(stz_s):
    """B_halo model from NE2001.
    B (stz vector field) ([muG], [muG], [muG]) as a function of stz ([kpc],
    [rad], [kpc]). Vectorized manually; batch dimension is the first.
    """
    R_s, t_s, z_s = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    BH0, RH0, zH0 = 10, 4, 1.5
    zH1_s = jnp.where(jnp.abs(z_s) < zH0, 0.2, 0.4)
    return jnp.stack([jnp.zeros_like(zH1_s),
                      BH0 / (1+((jnp.abs(z_s)-zH0)/zH1_s)**2)*R_s/RH0*jnp.exp(-(R_s-RH0)/RH0),
                      jnp.zeros_like(zH1_s)], axis=-1)

####################
## electron density

spec_ind_p = 3
spec_ind_alpha = (spec_ind_p+1)/2

def n_e(stz_s): # [unnorm] ( [[kpc], [1], [kpc]] ), vectorized
    """Electron density model from NE2001.
    n_e [unnormalized] as a function of stz ([kpc], [rad], [kpc]). Vectorized
    manually; batch dimension is the first.
    """
    s, t, z = stz_s[:,0], stz_s[:,1], stz_s[:,2]
    return jnp.where(jnp.abs(z)>1, 0.0,
                     jnp.exp(-(jnp.maximum(s,3)-r_Sun)/8-jnp.abs(z)))