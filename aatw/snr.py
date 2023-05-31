"""Functions and class for supernova remnants (SNR)"""

import sys
sys.path.append('..')

import numpy as np
from scipy import integrate
from astropy.coordinates import SkyCoord

import jax.numpy as jnp

from aatw.units_constants import *
from aatw.spectral import *


#===== image =====

def gaussian_val(sigma, x0, y0, x, y):
    """x, y can take any shape. Use np to be variable."""
    return np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

@jit
def gaussian_integral_estimate(x0, y0, sigma,
                               x_range, y_range, sample_n=(10, 10)):
    """Rough estimate of 2D gaussian integral in rectangles by sampling points
    uniformly.
    x0, y0, sigma: parameters for the gaussian.
    x/y_range: (x/y_start, x/y_end).
    sample_n: (# of pts sampled in x, ... in y)
    """
    x_edges = jnp.linspace(*x_range, sample_n[0]+1)
    y_edges = jnp.linspace(*y_range, sample_n[1]+1)
    xs = (x_edges[1:] + x_edges[:-1]) / 2
    ys = (y_edges[1:] + y_edges[:-1]) / 2
    x_grid, y_grid = jnp.meshgrid(xs, ys)
    x = x_grid.flatten()
    y = y_grid.flatten()
    z = jnp.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )
    return jnp.sum(z) * (x_range[1]-x_range[0]) * (y_range[1]-y_range[0]) \
           / (sample_n[0]*sample_n[1])

def norm_gaussian(sigma, x0=0, y0=0):
    """Returns a 2D gaussian function (normalized) as a function of sigma [rad]."""
    return lambda x, y: 1/(2*np.pi*sigma**2) * np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

def gaussian(sigma, x0=0, y0=0):
    """Returns a 2D gaussian function (unnormalized) as a function of sigma [rad]."""
    return lambda x, y: np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma**2) )

def add_image_to_map(fullmap, ra_s, dec_s, ra_edges, dec_edges, pixel_area_map,
                     source_ra, source_dec, image_sigma, n_sigma, T_submap_prefactor,
                     use_gaussian_intg=False, modify_mode='add', debug=False):
    
    if source_dec < dec_edges[0] or source_dec > dec_edges[-1]: # can't see source
        return
    
    i_ra_st  = int(np.searchsorted(ra_s,  source_ra  - n_sigma*image_sigma))
    i_ra_ed  = int(np.searchsorted(ra_s,  source_ra  + n_sigma*image_sigma))
    i_dec_st = int(np.searchsorted(dec_s, source_dec - n_sigma*image_sigma))
    i_dec_ed = int(np.searchsorted(dec_s, source_dec + n_sigma*image_sigma))
    if i_ra_ed == i_ra_st:
        if i_ra_ed == len(ra_s):
            i_ra_st -= 1
        else:
            i_ra_ed += 1
    if i_dec_ed == i_dec_st:
        if i_dec_ed == len(dec_s):
            i_dec_st -= 1
        else:
            i_dec_ed += 1

    ra_subgrid, dec_subgrid = np.meshgrid(ra_s[i_ra_st:i_ra_ed], dec_s[i_dec_st:i_dec_ed])

    if ra_subgrid.shape == (1, 1):
        tmpl = np.array([[1.]])
    else:
        if use_gaussian_intg:
            tmpl = np.zeros_like(ra_subgrid)
            for i_ra in range(i_ra_st, i_ra_ed):
                for i_dec in range(i_dec_st, i_dec_ed):
                    tmpl[i_dec-i_dec_st,i_ra-i_ra_st] = gaussian_integral_estimate(
                        source_ra, source_dec, image_sigma,
                        (ra_edges[i_ra],ra_edges[i_ra+1]), (dec_edges[i_dec],dec_edges[i_dec+1])
                    )
        else:
            tmpl = gaussian_val(image_sigma, source_ra, source_dec, ra_subgrid, dec_subgrid)
    pixel_area_submap = pixel_area_map[i_dec_st:i_dec_ed, i_ra_st:i_ra_ed]
    tmpl = np.array(tmpl) * np.cos(dec_s[i_dec_st:i_dec_ed])[:,None]# * pixel_area_submap
    tmpl /= np.sum(tmpl)

    #T_submap = snr.Sgnu(freq)*Jy * tmpl * c0**2 / (2 * freq**2 * pixel_area_submap * kb)
    #T_submap_prefactor = snr.Sgnu(freq)*Jy * 1 * c0**2 / (2 * freq**2 * 1 * kb)
    T_submap = T_submap_prefactor * tmpl / pixel_area_submap
    # [K] = [MHz^2 g] [cm MHz]^2 [MHz]^-2 [cm^2 MHz^2 g K^-1]^-1
    
    if modify_mode == 'add':
        fullmap[i_dec_st:i_dec_ed, i_ra_st:i_ra_ed] += T_submap
    elif modify_mode == 'replace':
        fullmap[i_dec_st:i_dec_ed, i_ra_st:i_ra_ed] = T_submap
    else:
        raise NotImplementedError(modify_mode)
        
    if debug:
        print(np.max(T_submap))


#===== SNR =====

class SNR:
    """Class for supernova remnants (SNR)."""
    
    def __init__(self, ID=None, name_alt=None, snr_type=None, l=None, b=None,
                 size=None, d=None, t_free=None, t_now=None, t_MFA=None,
                 si=None, Snu1GHz=None, snrcat_dict=None):
        self.ID = ID
        self.name_alt = name_alt
        self.snr_type = snr_type
        
        self.l = l # [rad] | galactic longitude
        self.b = b # [rad] | galactic latitude
        if self.l == 0 and self.b == 0:
            self.b = np.deg2rad(0.1)
        self.size = size # [rad] | size estimate
        self.d = d # [kpc] | distance estimate
        
        self.t_free = t_free # [yr] | t=0 at birth
        self.t_now  = t_now  # [yr] | t=0 at birth
        self.t_MFA  = t_MFA  # [yr] | onset time of magnetic field amplification
        self.si = si # [1] | si>0 | spectral index = (p-1)/2
        self.Snu1GHz = Snu1GHz # [Jy]
        
        self.snrcat_dict = snrcat_dict
        
    def build(self, rho_DM=None, tiop='2'):
        
        #========== variables ==========
        self.cl = self.l + np.pi - 2*np.pi*int(self.l>=np.pi) # [rad] | countersource l | [-pi,pi)
        self.cb = - self.b # [rad] | countersource b
        self.coord = SkyCoord(l=self.l, b=self.b, frame='galactic', unit='rad')
        self.ra  = self.coord.icrs.ra.rad
        self.dec = self.coord.icrs.dec.rad
        self.hemisph = 'N' if self.dec > 0 else 'S'
        self.thetaGCCS = np.arccos( -np.cos(self.l)*np.cos(self.b) ) # [rad] | GC and counter source
        self.thetaGCS = np.pi - self.thetaGCCS # [rad] | GC and source
        
        self.p   = 2*self.si+1 # si controls p
        self.ti1 = -2*(self.p+1)/5
        self.ti2 = -4*self.p/5
        
        #========== Snu ==========
        def Snu(nu):
            """Speciic flux [Jy] as a function of frequency nu [MHz]. nu can be
            a vector.
            """
            return self.Snu1GHz * (nu/1000)**(-self.si)
        self.Snu = Snu
        
        ti = self.ti1 if tiop=='1' else self.ti2
        def Snu_t_fl(nu, t):
            """Snu: constant --t_MFA--> t^ti. Doesn't care about t_free.
            units: [Jy]([MHz], [yr], ..).
            """
            if t < 0:
                return 0.
            elif t > self.t_MFA:
                return self.Snu(nu) * (t/self.t_now)**ti
            else:
                return self.Snu(nu) * (self.t_MFA/self.t_now)**ti
        self.Snu_t_fl = Snu_t_fl
        
        #========== gegenschein ==========
        nu_ref = 1000 # [MHz]
        intgd = lambda xp: self.Snu_t_fl(nu_ref, self.t(xp)) * rho_DM(np.maximum(self.Gr(xp), 0.01)) * kpc # converted intgd*dx from [Jy g/cm^3 kpc] to [Jy g/cm^3 cm] to bring numerical value closer to 1
        intg, err = integrate.quad(intgd, 0, self.xp(0)) # [Jy g/cm^2]
        self.Sgnu_ref = prefac(nu_ref) * intg # = [g^-1 cm^2] [Jy g cm^-2] = [Jy]
        def Sgnu(nu): # [Jy]([MHz])
            return self.Sgnu_ref * (prefac(nu)/prefac(nu_ref)) * (self.Snu(nu)/self.Snu1GHz)
        self.Sgnu = Sgnu
        
        imsz_intgd = lambda xp: self.image_sigma_at(xp) * self.Snu_t_fl(nu_ref, self.t(xp)) * rho_DM(self.Gr(xp)) * kpc
        imsz_intg, err = integrate.quad(imsz_intgd, 0, self.xp(0))
        self.image_sigma = imsz_intg/intg # [arcmin]
        
        #========== front gegenschein ==========
        nu_ref = 1000 # [MHz]
        intgd = lambda xp: self.Snu_t_fl(nu_ref, self.t_fg(xp)) * rho_DM(np.maximum(self.Gr_fg(xp), 0.01)) * kpc # converted intgd*dx from [Jy g/cm^3 kpc] to [Jy g/cm^3 cm] to bring numerical value closer to 1
        intg, err = integrate.quad(intgd, self.d, self.xp_fg(0)) # [Jy g/cm^2]
        self.Sfgnu_ref = prefac(nu_ref) * intg # = [g^-1 cm^2] [Jy g cm^-2] = [Jy]
        def Sfgnu(nu): # [Jy]([MHz])
            return self.Sfgnu_ref * (prefac(nu)/prefac(nu_ref)) * (self.Snu(nu)/self.Snu1GHz)
        self.Sfgnu = Sfgnu
        
        imsz_intgd = lambda xp: self.image_sigma_at_fg(xp) * self.Snu_t_fl(nu_ref, self.t_fg(xp)) * rho_DM(self.Gr_fg(xp)) * kpc
        imsz_intg, err = integrate.quad(imsz_intgd, self.d, self.xp_fg(0))
        self.image_sigma_fg = imsz_intg/intg # [arcmin]
        if np.isnan(imsz_intg):
            raise ValueError(f'{self.ID} image_sigma_fg')
        
        
    def name(self):
        """Returns name_alt or ID if name_alt is not present."""
        return self.name_alt if self.name_alt != '' else self.ID
    
    def size_t(self, t):
        """Size [rad] as a function of t [yr]."""
        if t > self.t_free:
            return self.size * (t/self.t_now)**0.4
        elif t > 0:
            return self.size * (self.t_free/self.t_now)**0.4 * (t/self.t_free)
        else:
            return 0.
        
    def xp(self, t):
        """x' (x_d) [kpc] as a function of SNR t [yr] (can be a vector)."""
        return c0_kpc_yr * (self.t_now-t) / 2
    
    def xp_fg(self, t):
        """x' (x_d) [kpc] as a function of SNR t [yr] (can be a vector).
        Front gegenschein."""
        return c0_kpc_yr * (self.t_now-t) / 2 + self.d
    
    def t(self, xp):
        """SNR t [yr] as a function of x' (x_d) [kpc] (can be a vector)."""
        return self.t_now - 2*xp/c0_kpc_yr
    
    def t_fg(self, xp):
        """SNR t [yr] as a function of x' (x_d) [kpc] (can be a vector).
        Front gegenschein."""
        return self.t_now - 2*(xp - self.d)/c0_kpc_yr
    
    def blur_sigma(self, xp):
        """Blur sigma [rad] as a function of x' (x_d) [kpc] (can be a vector)."""
        return (1 + xp/self.d) * 2 * sigmad_over_c
    
    def blur_sigma_fg(self, xp):
        """Blur sigma [rad] as a function of x' (x_d) [kpc] (can be a vector).
        Front gegenschein."""
        return (xp/self.d - 1) * 2 * sigmad_over_c
    
    def Gr(self, xp):
        """Distance to GC [kpc] as a function of x' (x_d) [kpc] (can be a vector)."""
        return jnp.sqrt(xp**2 + r_Sun**2 - 2*r_Sun*xp*jnp.cos(self.thetaGCCS))
    
    def Gr_fg(self, xp):
        """Distance to GC [kpc] as a function of x' (x_d) [kpc] (can be a vector).
        Front gegenschein."""
        return jnp.sqrt(xp**2 + r_Sun**2 - 2*r_Sun*xp*jnp.cos(self.thetaGCS))
    
    def image_sigma_at(self, xp):
        """Image sigma [rad] as a function of x' (x_d) [kpc]."""
        return np.sqrt((self.size_t(self.t(xp))/4)**2 + self.blur_sigma(xp)**2)
    
    def image_sigma_at_fg(self, xp):
        """Image sigma [rad] as a function of x' (x_d) [kpc].
        Front gegenschein."""
        return np.sqrt((self.size_t(self.t_fg(xp))/4)**2 + self.blur_sigma_fg(xp)**2)
        

#===== SNR utilities =====

def GID(l, b):
    if isinstance(l, str):
        l = float(l)
    if isinstance(b, str):
        sign = b[0]
        b = float(b)
    else:
        sign = '+' if b>0 else '-'
    return 'G' + ('%05.1f'%l) + sign + ('%04.1f'%np.abs(b))

def get_snr(name, snr_list):
    for snr in snr_list:
        if snr.ID==name or snr.name_alt==name:
            return snr
    return None