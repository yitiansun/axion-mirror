import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from scipy import interpolate

import matplotlib as mpl
import matplotlib.pyplot as plt

####################
## map functions

def antipodal_lb_map(lb_map):
    """Antipodal map of a lb_map with dimensions (~b, ~l)."""
    assert lb_map.shape[1] % 2 == 0
    return jnp.roll(jnp.flipud(lb_map), int(lb_map.shape[1]/2), axis=1)

def padded_interpolator(l, b, m):
    """Interpolator used to upsample l, b maps."""
    padded_b = np.zeros((len(b)+2,))
    padded_b[1:-1] = b
    padded_b[0] = b[0] - (b[1]-b[0])
    padded_b[-1] = b[-1] + (b[-1]-b[-2])
    
    padded_l = np.zeros((len(l)+2,))
    padded_l[1:-1] = l
    padded_l[0] = l[0] - (l[1]-l[0])
    padded_l[-1] = l[-1] + (l[-1]-l[-2])
    
    padded_m = np.zeros((len(b)+2,len(l)+2))
    padded_m[1:-1, 1:-1] = m
    padded_m[0,1:-1] = m[-1]
    padded_m[-1,1:-1] = m[0]
    padded_m[:,0] = padded_m[:,-2]
    padded_m[:,-1] = padded_m[:,1]
    
    return interpolate.interp2d(padded_l, padded_b, padded_m)

def interp2d(f, x0, x1, xv):
    """Interpolates f(x) at values in xvs. Does not do bound checks.
    f : (n>=2 D) array of function value.
    x0 : 1D array of input value, corresponding to first dimension of f.
    x1 : 1D array of input value, corresponding to second dimension of f.
    xv : [x0, x1] values to interpolate.
    """
    xv0, xv1 = xv
    
    li0 = jnp.searchsorted(x0, xv0) - 1
    lx0 = x0[li0]
    rx0 = x0[li0+1]
    p0 = (xv0-lx0) / (rx0-lx0)
    
    li1 = jnp.searchsorted(x1, xv1) - 1
    lx1 = x1[li1]
    rx1 = x1[li1+1]
    p1 = (xv1-lx1) / (rx1-lx1)
    
    fll = f[li0,li1]
    return fll + (f[li0+1,li1]-fll)*p0 + (f[li0,li1+1]-fll)*p1

interp2d_vmap = jit(vmap(interp2d, in_axes=(None, None, None, 0)))

def interpolate_padded(m, l, b, lb_s):
    
    padded_b = np.zeros((len(b)+2,))
    padded_b[1:-1] = b
    padded_b[0] = b[0] - (b[1]-b[0])
    padded_b[-1] = b[-1] + (b[-1]-b[-2])
    
    padded_l = np.zeros((len(l)+2,))
    padded_l[1:-1] = l
    padded_l[0] = l[0] - (l[1]-l[0])
    padded_l[-1] = l[-1] + (l[-1]-l[-2])
    
    padded_m = np.zeros((len(b)+2,len(l)+2))
    padded_m[1:-1, 1:-1] = m
    padded_m[0,1:-1] = m[-1]
    padded_m[-1,1:-1] = m[0]
    padded_m[:,0] = padded_m[:,-2]
    padded_m[:,-1] = padded_m[:,1]
    
    return interp2d_vmap(
        jnp.asarray(padded_m),
        jnp.asarray(padded_l),
        jnp.asarray(padded_b),
        lb_s
    )


####################
## plotting

def plot_radec(z, extent=None, vmax=None, vmin=None, log_norm=True, figsize=(8, 5), title='',
               save_fn=None, **imshow_kwargs):
    
    if vmax is None:
        vmax = jnp.max(z)
    if vmin is None:
        vmin = jnp.min(z)
    if log_norm:
        norm = mpl.colors.LogNorm(vmin, vmax)
    else:
        norm = mpl.colors.Normalize(vmin, vmax)
        
    default_kwargs = dict(
        extent = extent,
        cmap = 'magma',
        norm = norm,
    )
    default_kwargs.update(imshow_kwargs)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(jnp.flip(z), **default_kwargs)
    ax.set(aspect=1)
    ax.set(title=title, xlabel=r'ra [deg]', ylabel=r'dec [deg] (not uniform)')
    ax.set(xticks=jnp.linspace(extent[0], extent[1], 7),
           yticks=jnp.linspace(extent[2], extent[3], 2))
    fig.colorbar(im, ax=ax, shrink=0.8, orientation='horizontal', aspect=40)
    if save_fn is None:
        plt.show()
    else:
        plt.savefig(save_fn)
        #plt.close()
        print(f'Plot saved: {save_fn}')

def plot_lb(z, figsize=(8, 4), log_norm=True, title='', **imshow_kwargs):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    default_kwargs = dict(
        extent = (180,-180,-90,90),
        cmap = 'magma'
    )
    if log_norm:
        imshow_kwargs.update(dict(
            norm = mpl.colors.LogNorm(jnp.min(z), jnp.max(z))
        ))
    default_kwargs.update(imshow_kwargs)
    im = ax.imshow(jnp.flip(z), **default_kwargs)
    ax.set(aspect=1)
    ax.set(title=title, xlabel=r'$l$ [deg]', ylabel=r'$b$ [deg]')
    ax.set(xticks=jnp.linspace(180, -180, 7), yticks=jnp.linspace(-60, 60, 3))
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.show()


def plot_hv(func, smax=None, h_zval=0, zmax=None, v_tval=0, npix=None,
            title='', symm_color=False):
    
    x_s = jnp.linspace(-smax, smax, npix)
    y_s = jnp.linspace(-smax, smax, npix)
    x_ss, y_ss = jnp.meshgrid(x_s, y_s)
    
    x_in = jnp.reshape(x_ss, (npix*npix,))
    y_in = jnp.reshape(y_ss, (npix*npix,))
    stz_in = jnp.stack([jnp.sqrt(x_in**2+y_in**2),
                        jnp.arctan2(y_in, x_in),
                        jnp.full_like(x_in, h_zval)], axis=-1)
    hslice = jnp.reshape(func(stz_in), (npix, npix))
    
    s_s = jnp.linspace(0, smax, npix)
    z_s = jnp.linspace(-zmax, zmax, npix)
    s_ss, z_ss = jnp.meshgrid(s_s, z_s)
    
    s_in = jnp.reshape(s_ss, (npix*npix,))
    z_in = jnp.reshape(z_ss, (npix*npix,))
    stz_in = jnp.stack([s_in,
                        jnp.full_like(s_in, v_tval),
                        z_in], axis=-1)
    vslice = jnp.reshape(func(stz_in), (npix, npix))

    fig, axs = plt.subplots(1, 2, figsize=(12,6))

    vmax = jnp.max(jnp.array([jnp.max(hslice[hslice<jnp.inf]), jnp.max(vslice[vslice<jnp.inf])]))
    vmin = jnp.min(jnp.array([jnp.min(hslice[hslice<jnp.inf]), jnp.min(vslice[vslice<jnp.inf])]))
    if symm_color:
        vabs = jnp.max(jnp.array([jnp.abs(vmax), jnp.abs(vmin)]))
        kwargs = dict(vmin=-vabs, vmax=vabs, cmap='coolwarm')
    else:
        kwargs = dict(vmin=0, vmax=vmax, cmap='magma')
    
    im = axs[0].imshow(jnp.flipud(hslice), extent=(-smax, smax, -smax, smax), **kwargs)
    axs[1].imshow(jnp.flipud(jnp.array(vslice)), extent=(0, smax, -zmax, zmax), **kwargs)
    
    axs[0].set(xlabel='x [kpc]', ylabel='y [kpc]', title=title+f'$z={h_zval}$~kpc')
    axs[1].set(xlabel='x [kpc]', ylabel='z [kpc]', title=title+f'$\phi={jnp.rad2deg(v_tval)}^\circ$')
    fig.colorbar(im, ax=axs, orientation='horizontal', aspect=40)
    plt.show()