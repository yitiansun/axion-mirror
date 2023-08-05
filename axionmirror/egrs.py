"""Class and functions for extra-galactic radio sources"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy.io import fits


@dataclass
class EGRS:
    name: str = None
    l: float = None # [rad] | galactic longitude
    b: float = None # [rad] | galactic latitude
    ra: float = None # [rad]
    dec: float = None # [rad]
    spec: np.ndarray = None # [[freq, flux], [freq, flux], ...] must be sorted
    si: float = None
    min_extrap_freq: ClassVar[float] = 30 # [MHz]
    max_extrap_freq: ClassVar[float] = 1500 # [MHz]
    
    def __post_init__(self):
        self.spec = self.spec[np.argsort(self.spec[:,0])]

    @property
    def l_0center(self):
        return self.l if self.l < np.pi else self.l - 2*np.pi
    
    @property
    def min_data_freq(self):
        return self.spec[0,0]
    @property
    def max_data_freq(self):
        return self.spec[-1,0]
    @property
    def min_data_Snu(self):
        return self.spec[0,1]
    @property
    def max_data_Snu(self):
        return self.spec[-1,1]
        
    def Snu_no_extrap(self, nu):
        """
        Given frequency nu [MHz] returns flux density Snu [Jy].
        No extrapolation.
        """
        return np.interp(nu, self.spec[:,0], self.spec[:,1])

    def Snu(self, nu):
        """
        Given frequency nu [MHz] returns flux density Snu [Jy].
        With extrapolation.
        """
        if nu < self.min_data_freq:
            return self.min_data_Snu * (nu/self.min_data_freq) ** self.si
        elif nu > self.max_data_freq:
            return self.max_data_Snu * (nu/self.max_data_freq) ** self.si
        else:
            return self.Snu_no_extrap(nu)
        
    @property
    def lower_extrap_spec(self):
        """Extrapolation points below min_data_freq."""
        if self.min_data_freq < EGRS.min_extrap_freq:
            return None
        else:
            nu = EGRS.min_extrap_freq
            return np.array([[nu, self.Snu(nu)], self.spec[0]])
    @property
    def upper_extrap_spec(self):
        """Extrapolation points above max_data_freq."""
        if self.max_data_freq > EGRS.max_extrap_freq:
            return None
        else:
            nu = EGRS.max_extrap_freq
            return np.array([self.spec[-1], [nu, self.Snu(nu)]])
        
    def plot_spec(self, ax, interp_color='k', extrap_color='C1', alpha=1.):
        """Plot spectrum."""
        ax.plot(self.spec[:,0], self.spec[:,1], color=interp_color, alpha=alpha)
        lower_extrap_spec = self.lower_extrap_spec
        if lower_extrap_spec is not None:
            ax.plot(lower_extrap_spec[:,0], lower_extrap_spec[:,1], color=extrap_color, alpha=alpha)
        upper_extrap_spec = self.upper_extrap_spec
        if upper_extrap_spec is not None:
            ax.plot(upper_extrap_spec[:,0], upper_extrap_spec[:,1], color=extrap_color, alpha=alpha)
        
        
#===== utils =====

def egrs_list_keuhr():
    
    with fits.open("../data/egrs/keuhr_catalog.fits") as hdul:
        data = hdul[1].data
        
    egrs_list = []
    i_d = 0

    while i_d < len(data):
        d = data[i_d]
        name = d['NAME']
        i_d_end = i_d
        while i_d_end < len(data) and data[i_d_end]['NAME'] == name:
            i_d_end += 1

        egrs = EGRS(
            name = d['NAME'],
            l = np.deg2rad(d['LII']),
            b = np.deg2rad(d['BII']),
            ra = np.deg2rad(d['RA']),
            dec = np.deg2rad(d['DEC']),
            spec = np.stack([data[i_d:i_d_end]['FREQUENCY'],
                             data[i_d:i_d_end]['FLUX_RADIO']/1000.], axis=-1),
            si = d['SPECTRAL_INDEX'],
        )
        egrs_list.append(egrs)

        i_d = i_d_end
        
    return egrs_list