"""Class for telescopes"""

import sys
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

sys.path.append("..")
import axionmirror.units_constants as uc

data_dir = "../data"


@dataclass
class Telescope:
    """
    A dataclass for drift scan telescopes.

    Args:
        name (str)
        nu_min (float): Min frequency of telescope [MHz].
        nu_max (float): Max frequency of telescope [MHz].
        dec (float): Telescope declination.
        survey_za_max (float): Max zenith angle of survey in declination [rad].
        size_ra (float): Array size in right ascension [cm].
        size_dec (float): Array size in declination [cm].
        eta_f_ra (float): Filling factor in right ascension.
        eta_f_dec (float): Filling factor in declination.
        eta_a_ra (float): Aperture efficiency in right ascension.
        eta_a_dec (float): Aperture efficiency in declination.
        primary_beam_baseline_ra (float): Primary beam baseline in right ascension [cm].
        primary_beam_baseline_dec (float): Primary beam baseline in declination [cm].
        pointing (bool): Whether or not the telescope is pointing.

        eta_sig (float): Signal chain efficiency.
        T_sys_raw (float): Raw system temperature [K].
        t_obs_days (float): Observation time [day].
        double_pass_dec (Optional[float]): Optional parameter for double pass in declination.

    Note:
        All length measurements are in [cm], angles are in [rad], temperature is in [K], and time is in [day].
    """

    name: str
    nu_min: float
    nu_max: float
    dec: float
    survey_za_max: float
    size_ra: float
    size_dec: float
    eta_f_ra: float
    eta_f_dec: float
    eta_a_ra: float
    eta_a_dec: float
    primary_beam_baseline_ra: float
    primary_beam_baseline_dec: float
    pointing: bool

    eta_sig: float
    T_sys_raw: float
    t_obs_days: float
    
    def __post_init__(self):

        self.survey_ra_max = 2*np.pi
        self.survey_ra_min = 0
        self.survey_dec_max = self.dec + self.survey_za_max
        if self.survey_dec_max > np.pi/2:
            # if survey dec max is beyond the North Pole, set double pass dec
            self.double_pass_dec = np.pi - self.survey_dec_max
            self.survey_dec_max = np.pi/2
        else:
            self.double_pass_dec = None
        self.survey_dec_min = self.dec - self.survey_za_max
        if self.survey_dec_min < -np.pi/2:
            # should not have case where survey dec min is beyond the South Pole
            raise NotImplementedError
        
        self.extent = tuple(np.round(np.rad2deg(
            [self.survey_ra_max, self.survey_ra_min, self.survey_dec_min, self.survey_dec_max]
        )))
        self.T_sys = self.T_sys_raw / self.eta_sig
        
    def ra_pixel_size(self, nu, ra):
        """Pixel size in right ascension [rad]."""
        return (uc.c0 / nu) / (self.eta_f_ra * self.eta_a_ra * self.size_ra)
    
    def dec_pixel_size(self, nu, dec):
        """Pixel size in declination [rad]."""
        denom = self.eta_f_dec * self.eta_a_dec * self.size_dec * np.cos(dec)
        if not self.pointing:
            denom *= np.cos(dec - self.dec)
        return (uc.c0 / nu) / denom
            

#===== telescope instances =====

CHIME = Telescope(
    name = 'CHIME',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(49.3),
    survey_za_max = np.deg2rad(60.),
    size_ra = 86 * 100, size_dec = 78 * 100,
    eta_f_ra = 80/86, eta_f_dec = 1.,
    eta_a_ra = 0.5, eta_a_dec = 1.,
    primary_beam_baseline_ra = 20 * 100, primary_beam_baseline_dec = None,
    pointing = False,

    eta_sig = 1,
    T_sys_raw = 20, # with signal chain efficiency taken into account
    t_obs_days = 5 * 365.25,
)

CHORD = Telescope(
    name = 'CHORD',
    nu_min = 300, nu_max = 1500,
    dec = np.deg2rad(49.3),
    survey_za_max = np.deg2rad(30.),
    size_ra = 22 * 7 * 100, size_dec = 23 * 9 * 100,
    eta_f_ra = np.sqrt(6*6*np.pi / (4*7*9)), eta_f_dec = np.sqrt(6*6*np.pi / (4*7*9)),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    pointing = True,

    eta_sig = 1,
    T_sys_raw = ..., # with signal chain efficiency taken into account
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
    eta_a = 0.5,
    eta_sig = 1,
    T_sys_raw = 50,
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
    eta_a = 0.5,
    eta_sig = 1,
    T_sys_raw = 50,
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
    eta_a = 0.5,
    eta_sig = 1,
    T_sys_raw = 100,
    t_obs_days = 2 * 365.25,
)