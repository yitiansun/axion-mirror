import jax.numpy as jnp
from jax import jit, vmap

import sys
sys.path.append('..')
from utils.units_constants import *

##############################
## spectral analysis
fDelta = 0.721 # frequency domain cut associated with the above bandwidth (2.17)
def dnu(nu): # MHz(MHz)
    return 2.17 * sigmad_over_c * nu


##############################
## coordinates

def GCstz(lbd):
    """ Galactic center cylindrical coordinates (s, t, z) [L, rad, L] from Galactic coordinates with depth (l, b, d) [rad, rad, L]. Vectorized; batch dimension is the last dimension.
    """
    l, b, d = lbd
    x = d * jnp.cos(b) * jnp.cos(l) - r_Sun
    y = d * jnp.cos(b) * jnp.sin(l)
    z = d * jnp.sin(b)
    return jnp.array([ jnp.sqrt(x**2 + y**2),
                       jnp.arctan2(y, x),
                       z ])

def Glbd(stz):
    """ Galactic coordinates with depth (l, b, d) [rad, rad, L] from Galactic center cylindrical coordinates (s, t, z) [L, rad, L]. Vectorized; batch dimension is the last dimension.
    """
    s, t, z = stz
    x = s * jnp.cos(t)
    y = s * jnp.sin(t)
    # z = z
    return jnp.array([ jnp.arctan2(y, x+r_Sun),
                       jnp.arctan2(z, jnp.sqrt((x+r_Sun)**2 + y**2)),
                       jnp.sqrt((x+r_Sun)**2 + y**2 + z**2) ])

def Gr(lbd):
    """ Distance to galactic center r [kpc] as a function of Galactic coordinates with depth (l, b, d) [rad, rad, L]. Vectorized; batch dimension is the last dimension.
    """
    l, b, d = lbd
    x = d * jnp.cos(b) * jnp.cos(l) - r_Sun
    y = d * jnp.cos(b) * jnp.sin(l)
    z = d * jnp.sin(b)
    return jnp.sqrt(x**2 + y**2 + z**2)


##############################
## dark matter

rs_NFW = 16 # [kpc]
norm_NFW = 1/( (r_Sun/rs_NFW)*(1+r_Sun/rs_NFW)**2 )
intg_ds = jnp.concatenate((jnp.linspace(0, 20, 200, endpoint=False),
                           jnp.linspace(20, 200, 200, endpoint=False),
                           jnp.linspace(200, 1000, 100, endpoint=True)))

@jit
def rho_NFW(r):
    """ NFW halo density rho [GeV/cm^3] as a function of distance to galactic center r [kpc]. Vectorized. 
    """
    return 1/( (r/rs_NFW)*(1+r/rs_NFW)**2 ) / norm_NFW * 0.46

@jit
@vmap
def rho_integral(lb):
    """ NFW halo density integral [GeV kpc/cm^3] as a function of galactic coordinates (l, b) [rad, rad]. Vectorized; batch dimension is the first dimension.
    """
    l, b = lb
    lbd = jnp.array([ jnp.full_like(intg_ds, l),
                      jnp.full_like(intg_ds, b),
                      intg_ds ])
    rhos = rho_NFW(Gr(lbd))
    return jnp.trapz(rhos, intg_ds)


##############################
## image
def norm_gaussian(sigma, x0=0, y0=0): # [func](deg) | return a f(x, y)
    sigma_sq = sigma**2
    return lambda x, y: 1/(2*np.pi*sigma_sq) * np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma_sq) )

def gaussian(sigma, x0=0, y0=0): # [func](deg) | return a f(x, y)
    sigma_sq = sigma**2
    return lambda x, y: np.exp( -((x-x0)**2+(y-y0)**2)/(2*sigma_sq) )


##############################
## source
class SNR:
    def __init__(self, ID=None, name_alt=None, snr_type=None,
                 l=None, b=None, size=None, d=None,
                 tf=None, tn=None, tMFA=None, si=None, Snu1GHz=None,
                 snrcat_dict=None):
        self.ID = ID
        self.name_alt = name_alt # other names
        self.snr_type = snr_type
        
        self.l = l # deg | glon
        self.b = b # deg | glat
        self.size = size # arcmin | estimate
        self.d = d # kpc | estimate
        
        self.tf = tf # yr | t free | guess value 100, 300 
        self.tn = tn # yr | t now  | estimate
        self.tMFA = tMFA # yr | onset time of magnetic field amplification
        self.si = si # 1  | si>0 | spectral index = (p-1)/2
        self.Snu1GHz = Snu1GHz # Jy
        
        self.snrcat_dict = snrcat_dict
        
    def build(self):
        self.cl = self.l+180 if self.l<180 else self.l-180 # deg | countersource gal. longitude
        self.cb = -self.b   # deg | countersource gal. latitude
        self.coord = SkyCoord(l=self.l, b=self.b, frame='galactic', unit='deg')
        self.ra  = self.coord.icrs.ra.deg
        self.dec = self.coord.icrs.dec.deg
        self.hemisph = 'N' if self.dec>0 else 'S'
        self.thetaGCCS = np.arccos( -np.cos(self.l*np.pi/180)*np.cos(self.b*np.pi/180) )
        
        # si controls p
        self.p   = 2*self.si+1
        self.ti1 = -2*(self.p+1)/5
        self.ti2 = -4*self.p/5
        self.Snu = lambda nu: self.Snu1GHz * (nu/1000)**(-self.si) # Jy(MHz) | default S_nu
            
        def Snu_t_fl(nu, t, tiop='1'): # linear in free expansion
            ti = self.ti1 if tiop=='1' else self.ti2
            if t > self.tMFA: # free expansion and adiabatic
                return self.Snu(nu) * (t/self.tn)**ti
            else: # magnetic field turn on
                return self.Snu(nu) * (self.tMFA/self.tn)**ti
        self.Snu_t_fl = Snu_t_fl

        def Snu_t_fp(nu, t, tiop='1'): # conserved flux continues in free expansion
            # assume tf > tMFA
            ti = self.ti1 if tiop=='1' else self.ti2
            if t > self.tf: # adiabatic
                return self.Snu(nu) * (t/self.tn)**ti
            elif t > self.tMFA: # free expansion
                return self.Snu(nu) * (self.tf/self.tn)**ti * (t/self.tf)**(ti/0.4)
            else: # magnetic field turn on
                return (t/self.tMFA) * self.Snu(nu) * (self.tf/self.tn)**ti * (self.tMFA/self.tf)**(ti/0.4)
        self.Snu_t_fp = Snu_t_fp
        
        def Snu_t_fv(nu, t, tiop='1'): # B ~ v ~ const in free expansion
            # assume tf > tMFA
            ti = self.ti1 if tiop=='1' else self.ti2
            if t > self.tf: # adiabatic
                return self.Snu(nu) * (t/self.tn)**ti
            elif t > self.tMFA: # free expansion
                return self.Snu(nu) * (self.tf/self.tn)**ti * (t/self.tf)**(1-self.p)
            else: # magnetic field turn on
                return (t/self.tMFA) * self.Snu(nu) * (self.tf/self.tn)**ti * (self.tMFA/self.tf)**(1-self.p)
        self.Snu_t_fv = Snu_t_fv
        
    def name(self):
        return self.name_alt if self.name_alt != '' else self.ID
    
    def size_t(self, t): # arcmin(yr): size scaled back according to expansion
        if t > self.tf:
            return self.size * (t/self.tn)**0.4
        elif t > 0:
            tfsize = self.size * (self.tf/self.tn)**0.4
            return tfsize * (t/self.tf)
        else:
            return 0.
    def xp(self, t): # kpc(yr): dDM as a function of tSNR
        return c0kpcyr * (self.tn-t) / 2
    def t(self, xp): # yr(kpc): inverse function
        return self.tn - 2*xp/c0kpcyr
    def blur_sigma(self, xp): # arcmin(kpc)
        return (1+xp/self.d) * (2*sigmad_over_c*(180/np.pi)) * 60
    def Gr(self, xp): # kpc(kpc): r to GC as function of DM xp and source
        return np.sqrt(xp**2 + Rs**2 - 2*Rs*xp*np.cos(self.thetaGCCS))
    def image_sigma_at(self, xp): # arcmin(kpc)
        return np.sqrt((self.size_t(self.t(xp))/4)**2 + self.blur_sigma(xp)**2)
        
        
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


##############################
# telescope

class Telescope:
    def __init__(self, name=None, hemisphere=None, nu_low=None, nu_high=None,
                 Aeff=None, sens=None, tobs=None, Trec=None,
                 sq_r=None, effi_eval=None):
        self.name = name
        self.hemisphere = hemisphere
        
        self.nu_low = nu_low   # MHz | frequency lower bound
        self.nu_high = nu_high # MHz | frequency upper bound
        
        self.Aeff = Aeff # cm^2   | Aeff(nu=None) | effective area
        self.sens = sens # cm^2/K | sens(nu=None) | sensitivity
        self.tobs = tobs # MHz^-1 | obsering time
        self.Trec = Trec # K      | receiver temperature
        
        self.sq_r = sq_r # deg | size of square that encloses beam
        self.effi_eval = effi_eval # func(func) | effi_eval(nenvl) | efficiency evaluator


##############################
## haslam

# WDIR = '/Users/sunyitian/Dropbox (MIT)/Documents/P/axion gegenschein/axionsversustheworld/'
# haslam_fitsfn = 'haslam_skymap_YS/lambda_haslam408_nofilt.fits'
# haslam_hdu = fits.open(WDIR+haslam_fitsfn)[1]
# haslam_hdu.header['COORDSYS'] = 'G'

# def haslamGXYZ(l=0, b=0, l_span=10, b_span=10, l_npix=100, b_npix=100): # deg # negative span to invert
#     target_header = fits.Header(
#         {'NAXIS' : 2, 'NAXIS1': l_npix, 'NAXIS2': b_npix,
#          'CTYPE1': 'GLON' , 'CUNIT1': 'deg', 'CRVAL1': l, 'CRPIX1': l_npix/2, 'CDELT1': l_span/l_npix,
#          'CTYPE2': 'GLAT', 'CUNIT2': 'deg', 'CRVAL2': b, 'CRPIX2': b_npix/2, 'CDELT2': b_span/b_npix,
#          'COORDSYS': 'G'}
#     )
#     Z, footprint = reproject_from_healpix(haslam_hdu, target_header)
#     X, Y = np.meshgrid(np.linspace(l-l_span/2, l+l_span/2, num=l_npix),
#                        np.linspace(b-b_span/2, b+b_span/2, num=b_npix))
#     return X, Y, Z/1000 # mK to K

# nu_beta = 2.5 # 1 | galactic synchrontron spectral index

# def Tbkg408(snr, sq_r=None): # K(<SNR>, deg)
#     _, _, Zbkg = haslamGXYZ(l=snr.cl, b=snr.cb,
#                             l_span=2*sq_r, b_span=2*sq_r,
#                             l_npix=10, b_npix=10)
#     return np.mean(Zbkg)