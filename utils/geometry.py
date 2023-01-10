import jax.numpy as jnp
from jax import jit, vmap

import sys
sys.path.append('..')
from utils.units_constants import *

def Gr(lbd):
    """ Distance to galactic center r [L] as a function of galactic coordinates
    with depth (l, b, d) [rad, rad, L]. Vectorized manually; batch dimension is
    the first dimension. """
    l, b, d = lbd[:, 0], lbd[:, 1], lbd[:, 2]
    x = d * jnp.cos(b) * jnp.cos(l) - r_Sun
    y = d * jnp.cos(b) * jnp.sin(l)
    z = d * jnp.sin(b)
    return jnp.sqrt(x**2 + y**2 + z**2)


########################################
##     below is old code
########################################

####################
## geometry

# batch dimension is the first dimension

# norm = vmap(jnp.linalg.norm())

def unit_vec(v_s):
    """Unit vectors of array of (xyz) vector. Batch dimension is first."""
    return v_s / vmap(jnp.linalg.norm(v_s))

def cross_product(v1xyz, v2xyz):
    v1x, v1y, v1z = v1xyz
    v2x, v2y, v2z = v2xyz
    return jnp.array([v1y*v2z - v2y*v1z,
                      v1z*v2x - v2z*v1x,
                      v1x*v2y - v2x*v1y])


####################
## coordinates

def GCstz(lbd):
    """Galactic center cylindrical coordinates (s, t, z) [L, rad, L] from
    Galactic coordinates with depth (l, b, d) [rad, rad, L]. Vectorized; batch
    dimension is the last dimension.
    """
    l, b, d = lbd
    x = d * jnp.cos(b) * jnp.cos(l) - r_Sun
    y = d * jnp.cos(b) * jnp.sin(l)
    z = d * jnp.sin(b)
    return jnp.array([ jnp.sqrt(x**2 + y**2),
                       jnp.arctan2(y, x),
                       z ])

def Glbd(stz):
    """Galactic coordinates with depth (l, b, d) [rad, rad, L] from Galactic
    center cylindrical coordinates (s, t, z) [L, rad, L]. Vectorized; batch
    dimension is the last dimension.
    """
    s, t, z = stz
    x = s * jnp.cos(t)
    y = s * jnp.sin(t)
    # z = z
    return jnp.array([ jnp.arctan2(y, x+r_Sun),
                       jnp.arctan2(z, jnp.sqrt((x+r_Sun)**2 + y**2)),
                       jnp.sqrt((x+r_Sun)**2 + y**2 + z**2) ])



def GCxyz_stz(stz):
    """Galactic center cartesian coordinates (x, y, z) [L, L, L] from Galactic
    center cylindrical coordinates (s, t, z) [L, rad, L]. Vectorized; batch
    dimension is the last dimension."""
    s, t, z = stz
    x = s * jnp.cos(t)
    y = s * jnp.sin(t)
    # z = z
    return jnp.array([x, y, z])

def los_direction(xyz):
    """Line of sight direction (radially outward) of xyz coordinates, in xyz
    coordinates. [kpc] Vectorized; batch dimension is the last dimension."""
    x, dy, dz = xyz
    dx = x + r_Sun
    return unit_vec(jnp.array([dx, dy, dz]))

def vstz2vxyz_stz(vsvtvz, stz):
    vs, vt, vz = vsvtvz
    s, t, z = stz
    return jnp.array([vs*jnp.cos(t) - vt*jnp.sin(t),
                      vs*jnp.sin(t) + vt*jnp.cos(t),
                      vz])