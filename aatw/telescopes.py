"""Class for telescopes"""

from dataclasses import dataclass
from typing import Union, Optional

import numpy as np


data_dir = "../data"


@dataclass
class Telescope:
    """Class for drift scan telescopes.
    Units:
        nu [MHz]
        lengths [cm]
        angles [rad]
        T [K]
        t [day]
    """
    name: str
    nu_min: float
    nu_max: float
    size_ra: float
    size_dec: float
    ra_max: float
    ra_min: float
    dec_max: float
    dec_min: float
    dec: float
    primary_beam_baseline: float
    eta: Union[str, float]
    T_sys: Union[str, float]
    t_obs_days: float
    double_pass_dec: Optional[float] = None
    
    def __post_init__(self):
        
        self.extent = (
            360, 0, np.round(np.rad2deg(self.dec_min)), np.round(np.rad2deg(self.dec_max))
        )
        
        if isinstance(self.eta, str):
            self.eta_data = np.loadtxt(self.eta)
            self.eta = lambda nu: np.interp(nu, self.eta_data[0], self.eta_data[1])
            
        if isinstance(self.T_sys, str):
            self.T_sys_data = np.loadtxt(self.T_sys)
            self.T_sys = lambda nu: np.interp(nu, self.T_sys_data[0], self.T_sys_data[1])
            

#===== telescope instances =====

CHIME = Telescope(
    name = 'CHIME',
    nu_min = 400, nu_max = 800,
    size_ra = 80 * 100, size_dec = 100 * 100,
    ra_max = 2*np.pi, ra_min = 0,
    dec_max = np.deg2rad(90), dec_min = np.deg2rad(-20),
    dec = np.deg2rad(49.3),
    primary_beam_baseline = 20 * 100,
    eta = 0.87,
    T_sys = 43,
    t_obs_days = 5 * 365.25,
    double_pass_dec = np.deg2rad(70),
)

CHIME_at_HIRAX = Telescope(
    name = 'CHIME_at_HIRAX',
    nu_min = 400, nu_max = 800,
    size_ra = 80 * 100, size_dec = 100 * 100,
    ra_max = 2*np.pi, ra_min = 0,
    dec_max = np.deg2rad(40), dec_min = np.deg2rad(-90),
    dec = np.deg2rad(-30.72),
    primary_beam_baseline = 20 * 100,
    eta = 0.87,
    T_sys = 43,
    t_obs_days = 5 * 365.25,
)

HIRAX_256 = Telescope(
    name = 'HIRAX-256',
    nu_min = 400, nu_max = 800,
    size_ra = 16 * 7 * 100, size_dec = 16 * 7 * 100,
    ra_max = 2*np.pi, ra_min = 0,
    dec_max = np.deg2rad(0), dec_min = np.deg2rad(-60),
    dec = np.deg2rad(-30.72),
    primary_beam_baseline = 6 * 100,
    eta = 1,
    T_sys = 50,
    t_obs_days = 2 * 365.25,
)

HIRAX_1024 = Telescope(
    name = 'HIRAX-1024',
    nu_min = 400, nu_max = 800,
    size_ra = 32 * 7 * 100, size_dec = 32 * 7 * 100,
    ra_max = 2*np.pi, ra_min = 0,
    dec_max = np.deg2rad(0), dec_min = np.deg2rad(-60),
    dec = np.deg2rad(-30.72),
    primary_beam_baseline = 6 * 100,
    eta = 1,
    T_sys = 50,
    t_obs_days = 2 * 365.25,
)

HERA = Telescope(
    name = 'HERA',
    nu_min = 50, nu_max = 250,
    size_ra = 18 * 14 * 100, size_dec = 18 * 14 * 100,
    ra_max = 2*np.pi, ra_min = 0,
    dec_max = np.deg2rad(0), dec_min = np.deg2rad(-60),
    dec = np.deg2rad(-30.72),
    primary_beam_baseline = 14 * 100,
    eta = 1,
    T_sys = 100,
    t_obs_days = 2 * 365.25,
)