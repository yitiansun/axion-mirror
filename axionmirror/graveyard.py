"""Functions or graveyard SNRs"""

import sys
sys.path.append("..")

from tqdm import tqdm

import numpy as np
from scipy import stats
import jax.numpy as jnp
from jax import vmap

from axionmirror.units_constants import *
from axionmirror.geometry import Glbd, GCstz
from axionmirror.stats import sample_from_pdf, poisson_process
from axionmirror.nfw import rho_NFW
from axionmirror.snr import SNR


#===== rate =====

def snr_forming_rate_CC(value='est'):
    """CC SNR formation rate [1/yr].
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    3.2+7.3-2.6 / 100yr
    """
    if value == 'est':
        return 3.2 / 100
    elif value == 'upper':
        return (3.2+7.3) / 100
    elif value == 'lower':
        return (3.2-2.6) / 100
    else:
        raise ValueError(value)

def snr_forming_rate_Ia(value='est'):
    """Ia SNR formation rate [1/yr].
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    1.4+1.4-0.8 / 100 yr
    """
    if value == 'est':
        return 1.4 / 100
    elif value == 'upper':
        return (1.4+1.4) / 100
    elif value == 'lower':
        return (1.4-0.8) / 100
    else:
        raise ValueError(value)

def snr_forming_rate_tot(value='est'):
    """Total SNR formation rate [1/yr].
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    4.6+7.4-2.7 / 100 yr
    """
    if value == 'est':
        return 4.6 / 100
    elif value == 'upper':
        return (4.6+7.4) / 100
    elif value == 'lower':
        return (4.6-2.7) / 100
    else:
        raise ValueError(value)
        

#===== location =====

def snr_stz_pdf_AKBVS(stz):
    """CC SNR location pdf.
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    Vectorized in first dimension.
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    s0 = 2.9 # [kpc]
    z0 = 0.095 # [kpc]
    return np.exp(-s/s0) * np.exp(-np.abs(z)/z0)

def sample_snr_stz_AKBVS(num_samples=1):
    """SNR location sampler.
    From 1306.0559 Adams Kochanek Beacom Vagins Stanek.
    """
    s0 = 2.9 # [kpc]
    s_pdf = lambda s: np.exp(-s/s0)
    s_start, s_end = 0, 15 # [kpc]
    
    z0 = 0.095 # [kpc]
    z_pdf = lambda z: np.exp(-np.abs(z)/z0)
    z_start, z_end = -1, 1 # [kpc]
    
    s_samples = sample_from_pdf(s_pdf, s_start, s_end, num_samples)
    z_samples = sample_from_pdf(z_pdf, z_start, z_end, num_samples)
    t_samples = np.random.uniform(0, 2*np.pi, num_samples)
    
    return np.stack([s_samples,
                     t_samples,
                     z_samples], axis=-1)

def snr_stz_pdf_G(stz):
    """SNR location pdf.
    Following 1508.02931 Green.
    Vectorized in first dimension.
    
    Take z to be Uniform(-0.1, 0.1).
    SNR type: no mention
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    
    R_sun = 8.5 # [kpc]
    alpha = 1.09 # [1]
    beta = 3.87 # [1]
    return (s/R_sun)**alpha * jnp.exp(-beta*(s-R_sun)/R_sun) * (jnp.abs(z) < 0.1)

def snr_stz_pdf_G_softz(stz):
    """Same as above except distribution on z is replace with exponential falloff.
    This is for SNR distance sampling (to avoid sampling outside of the support of pdf).
    """
    s, t, z = stz[:,0], stz[:,1], stz[:,2]
    
    R_sun = 8.5 # [kpc]
    alpha = 1.09 # [1]
    beta = 3.87 # [1]
    return (s/R_sun)**alpha * jnp.exp(-beta*(s-R_sun)/R_sun) * jnp.exp(-jnp.abs(z)/0.1)

def sample_snr_stz_G(num_samples=1):
    """SNR location sampler.
    Following 1508.02931 Green.
    Take z to be Uniform(-0.1, 0.1).
    """
    R_sun = 8.5 # [kpc]
    alpha = 1.09 # [1]
    beta = 3.87 # [1]
    s_pdf = lambda s: (s/R_sun)**alpha * np.exp(-beta*(s-R_sun)/R_sun)
    s_start, s_end = 0, 15 # [kpc]
    
    s_samples = sample_from_pdf(s_pdf, s_start, s_end, num_samples)
    z_samples = np.random.uniform(-0.1, 0.1, num_samples)
    t_samples = np.random.uniform(0, 2*np.pi, num_samples)
    
    return np.stack([s_samples,
                     t_samples,
                     z_samples], axis=-1)

def sample_snr_d(stz_pdf_func, l, b, lowerbound, upperbound, num_samples=1):
    """SNR distance sampler.
    Given stz_pdf_func(callable), l [rad], b [rad], lowerbound [kpc], upperbound [kpc],
    sample 1 instance of d [kpc].
    """
    def d_pdf(d):
        stz = GCstz(jnp.array([[l, b, d]]))
        return stz_pdf_func(stz)[0]
    
    return sample_from_pdf(vmap(d_pdf), lowerbound, upperbound, num_samples)


#===== spectral index =====

def sample_si(num_samples=1):
    """See ../notebooks/snr_graveyard.ipynb"""
    si_skewness, si_loc, si_scale = -1.2377486349155773, 0.5991927451179504, 0.1894835347218209
    return stats.skewnorm.rvs(si_skewness, loc=si_loc, scale=si_scale, size=num_samples)


#===== size =====

def sample_size_1kyr(num_samples=1):
    """See ../notebooks/snr_graveyard.ipynb"""
    size_skewness, size_loc, size_scale = 9.383925762403216, 0.0036672822153501466, 0.010441241123269699
    return stats.skewnorm.rvs(size_skewness, loc=size_loc, scale=size_scale, size=num_samples)

def sample_size_now(num_samples=1, t_now=...):
    """t_now in [yr], can be a vector."""
    return sample_size_1kyr(num_samples) * (1000/t_now)**0.4


#===== lightcurve =====

def sample_t_pk(num_samples=1):
    
    t_pk_mean = 50 # [day]
    t_pk_stddex = 0.9 # [1]
    return 10**(stats.norm.rvs(loc=np.log10(t_pk_mean), scale=t_pk_stddex, size=num_samples)) / 365.25 # [yr]

def sample_L_pk(num_samples=1):
    
    L_pk_mean = 3e25 # [erg/s/Hz]
    L_pk_stddex = 1.5 # [1]
    return 10**(stats.norm.rvs(loc=np.log10(L_pk_mean), scale=L_pk_stddex, size=num_samples))

def sample_Snu1GHz_pk(num_samples=1, si=..., d=...):
    """d in [kpc]. d and si can be vectors."""
    Snu6p3GHz_pk = sample_L_pk(num_samples) / (4*np.pi*(d*kpc)**2) / sec**2 / Jy # [Jy]
    return Snu6p3GHz_pk * np.sqrt(4*10) ** si
    
    
#===== t_now (age) =====

def sample_t_now(num_samples=1):
    """See ../notebooks/snr_graveyard.ipynb"""
    log10age_skewness, log10age_loc, log10age_scale = 0.9475698171302935, 3.5585998647723156, 0.6187394571802789
    return 10**stats.skewnorm.rvs(log10age_skewness, loc=log10age_loc, scale=log10age_scale, size=num_samples)


#===== t_free =====

def sample_t_free(num_samples=1):
    """See ../notebooks/snr_graveyard.ipynb"""
    t_free_skewness, t_free_loc, t_free_scale = -0.4409101842885288, 2.322143632416538, 0.8307012289498359
    return 10**stats.skewnorm.rvs(t_free_skewness, loc=t_free_loc, scale=t_free_scale, size=num_samples)

def fixed_t_free(value='est'):
    """[yr]"""
    if value == 'est':
        return 300.
    elif value == 'upper':
        return 100.
    elif value == 'lower':
        return 1000.
    else:
        raise ValueError(value)
    

#===== full graveyard sample =====
    
def sample_graveyard_snrs(t_cutoff=100000, verbose=1):
    """t_cutoff in [yr], set at end of adiabatic phase typically."""

    t_now_arr = np.array(poisson_process(snr_forming_rate_tot(), t_cutoff))
    n_snr = len(t_now_arr)
    if verbose >= 1:
        print(f'n_snr={n_snr}')
    
    lbd_arr = np.array(Glbd(sample_snr_stz_G(n_snr)), dtype=np.float64)
    si_arr = sample_si(n_snr)
    size_arr = sample_size_now(n_snr, t_now=t_now_arr)
    L_pk_arr = sample_L_pk(n_snr)
    t_pk_arr = sample_t_pk(n_snr)
    #t_free_arr = sample_t_free(n_snr)
    t_free_arr = np.full((n_snr,), fixed_t_free())
    Snu1GHz_pk_arr = sample_Snu1GHz_pk(n_snr, si=si_arr, d=lbd_arr[:, 2]),
    
    snr_list = []
    for i in (tqdm(range(n_snr)) if verbose >= 1 else range(n_snr)):
        
        snr = SNR(
            ID = f'Graveyard-{i}',
            l = lbd_arr[i, 0],
            b = lbd_arr[i, 1],
            d = lbd_arr[i, 2],
            size = size_arr[i],
            t_now = t_now_arr[i],
            t_free = t_free_arr[i],
            t_pk = t_pk_arr[i],
            Snu1GHz_pk = Snu1GHz_pk_arr[i],
            si = si_arr[i],
        )
        snr.build(rho_DM=rho_NFW, use_lightcurve=True, integrate_method='trapz')
        snr_list.append(snr)
    return snr_list