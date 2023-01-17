import jax.numpy as jnp
from jax import jit, vmap

import sys
sys.path.append('..')
from utils.units_constants import *


####################
## geometry

def unit_vec(v):
    """Unit vector(s) of xyz vector(s). Can take vector input; batch dimension
    is the first dimension.
    """
    return v / jnp.expand_dims(jnp.linalg.norm(v, axis=-1), axis=1)

def cross_product(xyz1, xyz2):
    """Cross product of two arrays xyz vectors. Vectorized manually; batch
    dimension is the first dimension.
    """
    x1, y1, z1 = xyz1[:, 0], xyz1[:, 1], xyz1[:, 2]
    x2, y2, z2 = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
    return jnp.stack([y1*z2 - y2*z1,
                      z1*x2 - z2*x1,
                      x1*y2 - x2*y1], axis=-1)


####################
## coordinates

def Gr(lbd):
    """Distance to galactic center r [L] as a function of galactic coordinates
    with depth (l, b, d) [rad, rad, L]. Vectorized manually; batch dimension is
    the first dimension.
    """
    l, b, d = lbd[:, 0], lbd[:, 1], lbd[:, 2]
    x = d * jnp.cos(b) * jnp.cos(l) - r_Sun
    y = d * jnp.cos(b) * jnp.sin(l)
    z = d * jnp.sin(b)
    return jnp.sqrt(x**2 + y**2 + z**2)


def GCstz(lbd):
    """Galactic center cylindrical coordinates (s, t, z) [L, rad, L] from
    Galactic coordinates with depth (l, b, d) [rad, rad, L]. Vectorized manually
    ; batch dimension is the first dimension.
    """
    l, b, d = lbd[:,0], lbd[:,1], lbd[:,2]
    x = d * jnp.cos(b) * jnp.cos(l) - r_Sun
    y = d * jnp.cos(b) * jnp.sin(l)
    z = d * jnp.sin(b)
    return jnp.stack([jnp.sqrt(x**2 + y**2),
                      jnp.arctan2(y, x),
                      z], axis=-1)

def Glbd(stz):
    """Galactic coordinates with depth (l, b, d) [rad, rad, L] from Galactic
    center cylindrical coordinates (s, t, z) [L, rad, L]. Vectorized manually;
    batch dimension is the first dimension.
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    x = s * jnp.cos(t)
    y = s * jnp.sin(t)
    # z = z
    return jnp.stack([jnp.arctan2(y, x+r_Sun),
                      jnp.arctan2(z, jnp.sqrt((x+r_Sun)**2 + y**2)),
                      jnp.sqrt((x+r_Sun)**2 + y**2 + z**2)], axis=-1)

def GCxyz_stz(stz):
    """Galactic center cartesian coordinates (x, y, z) [L, L, L] from Galactic
    center cylindrical coordinates (s, t, z) [L, rad, L]. Vectorized manually;
    batch dimension is the first dimension.
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    x = s * jnp.cos(t)
    y = s * jnp.sin(t)
    # z = z
    return jnp.stack([x, y, z], axis=-1)

def LOS_direction(xyz):
    """Line of sight direction (radially outward) of xyz coordinates, in xyz
    coordinates. [kpc] Vectorized manually; batch dimension is the first
    dimension.
    """
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    return unit_vec(jnp.stack([x+r_Sun, y, z], axis=-1))

def vstz2vxyz_stz(vstz, stz):
    """Converts a stz vector field in stz coordinates to a xyz vector field in 
    stz coordinates. Vectorized manually; batch dimension is the first
    dimension.
    """
    vs, vt, vz = vstz[:,0], vstz[:,1], vstz[:,2]
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    return jnp.stack([vs*jnp.cos(t) - vt*jnp.sin(t),
                      vs*jnp.sin(t) + vt*jnp.cos(t),
                      vz], axis=-1)