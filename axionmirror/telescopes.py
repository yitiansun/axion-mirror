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
        pointing (bool): Whether or not the telescope is pointing.
        survey_ra_max (float): Max right ascension of survey [rad].
        survey_ra_min (float): Min right ascension of survey [rad].
        fixed_survey_za_max (float): Fixed max zenith angle of survey in declination [rad]. If None, use half primary beam (dec) size.
        size_ra (float): Array size in right ascension [cm].
        size_dec (float): Array size in declination [cm].
        eta_f_ra (float): Filling factor in right ascension.
        eta_f_dec (float): Filling factor in declination.
        eta_a_ra (float): Aperture efficiency in right ascension.
        eta_a_dec (float): Aperture efficiency in declination.
        primary_beam_baseline_ra (float): Primary beam baseline in right ascension [cm].
        primary_beam_baseline_dec (float): Primary beam baseline in declination [cm].
        eta_sig (float): Signal chain efficiency.
        T_rec_raw (float): Raw receiver temperature [K].
        t_obs_days (float): Observation time [day].

    Note:
        Default length measurements are in [cm], angles are in [rad], temperature is in [K].
    """

    name: str
    nu_min: float
    nu_max: float
    dec: float
    pointing: bool
    survey_ra_max: float
    survey_ra_min: float
    fixed_survey_za_max: float
    size_ra: float
    size_dec: float
    eta_f_ra: float
    eta_f_dec: float
    eta_a_ra: float
    eta_a_dec: float
    primary_beam_baseline_ra: float
    primary_beam_baseline_dec: float
    eta_sig: float
    T_rec_raw: float
    t_obs_days: float

        
    def ra_pixel_size(self, nu, ra):
        """Pixel size in right ascension [rad]."""
        return (uc.c0 / nu) / (self.eta_f_ra * self.eta_a_ra * self.size_ra)
    
    def dec_pixel_size(self, nu, dec):
        """Pixel size in declination [rad]."""
        denom = self.eta_f_dec * self.eta_a_dec * self.size_dec * np.cos(dec)
        if not self.pointing:
            denom *= np.cos(dec - self.dec)
        return (uc.c0 / nu) / denom
    
    def primary_beam_size_ra(self, nu):
        """Primary beam size in right ascension direction (but not in right ascension coordinates) [rad]."""
        return (uc.c0 / nu) / (self.eta_a_ra * self.primary_beam_baseline_ra)
    
    def primary_beam_size_dec(self, nu):
        """Primary beam size in declination [rad]."""
        return (uc.c0 / nu) / (self.eta_a_dec * self.primary_beam_baseline_dec)
    
    def survey_za_max(self, nu):
        """Maximum survey zenith angle [rad]."""
        if self.fixed_survey_za_max is not None:
            return self.fixed_survey_za_max
        else:
            if self.pointing:
                raise NotImplementedError
            return self.primary_beam_size_dec(nu) / 2
        
    def survey_dec_max(self, nu):
        """Maximum survey declination [rad]."""
        return np.minimum(self.dec + self.survey_za_max(nu), np.pi/2)
    
    def double_pass_dec(self, nu):
        """Declination beyond which points on the sky enters FOV twice per day [rad]."""
        survey_dec_max = self.dec + self.survey_za_max(nu)
        if survey_dec_max > np.pi/2:
            return np.pi - survey_dec_max
        else:
            return None
        
    def survey_dec_min(self, nu):
        """Minimum survey declination [rad]."""
        survey_dec_min = self.dec - self.survey_za_max(nu)
        if survey_dec_min < -np.pi/2:
            raise NotImplementedError
        return np.maximum(survey_dec_min, -np.pi/2)
    
    def extent(self, nu):
        """Extent of survey in [deg]."""
        return tuple(np.rad2deg(
            [self.survey_ra_max, self.survey_ra_min, self.survey_dec_min(nu), self.survey_dec_max(nu)]
        ))
    
    @property
    def T_rec(self):
        """Receiver temperature [K]."""
        return self.T_rec_raw / self.eta_sig
    
    def t_obs(self, nu, dec):
        """Total observation time [s]. dec can take any shape."""
        t_per_day = 86400 # [s]
        if self.pointing:
            t_per_day *= self.primary_beam_size_dec(nu) / (self.survey_dec_max(nu) - self.survey_dec_min(nu))
        t_obs_per_day = t_per_day * self.primary_beam_size_ra(nu) / (2*np.pi * np.cos(dec))
        if self.double_pass_dec(nu) is not None:
            t_obs_per_day *= ((dec > self.double_pass_dec(nu)) + 1)
        
        return self.t_obs_days * t_obs_per_day
            

#===== telescope instances =====

CHIME = Telescope(
    name = 'CHIME',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(49.3),
    pointing = False,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(60.),
    size_ra = 86 * 100, size_dec = 78 * 100,
    eta_f_ra = 80/86, eta_f_dec = 1.,
    eta_a_ra = 0.5, eta_a_dec = 1.,
    primary_beam_baseline_ra = 20 * 100, primary_beam_baseline_dec = None,
    eta_sig = 1.,
    T_rec_raw = 20.,
    t_obs_days = 5 * 365.25,
)

HERA = Telescope(
    name = 'HERA',
    nu_min = 50, nu_max = 250,
    dec = np.deg2rad(-30.7),
    pointing = False,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = None, # use primary beam size (dec) / 2
    size_ra = 243 * 100, size_dec = 243 * 100,
    eta_f_ra = np.sqrt(0.834), eta_f_dec = np.sqrt(0.834),
    eta_a_ra = np.sqrt(0.6), eta_a_dec = np.sqrt(0.6),
    primary_beam_baseline_ra = 14 * 100, primary_beam_baseline_dec = 14 * 100,
    eta_sig = 1.,
    T_rec_raw = 100.,
    t_obs_days = 5 * 365.25,
)

CHORD = Telescope(
    name = 'CHORD',
    nu_min = 300, nu_max = 1500,
    dec = np.deg2rad(49.3),
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(30.),
    size_ra = 22 * 7 * 100, size_dec = 23 * 9 * 100,
    eta_f_ra = np.sqrt(0.45), eta_f_dec = np.sqrt(0.45),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = 1.,
    T_rec_raw = 30.,
    t_obs_days = 5 * 365.25,
)

HIRAX256 = Telescope(
    name = 'HIRAX256',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(-30.7),
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(30.),
    size_ra = 16 * 7 * 100, size_dec = 16 * 9 * 100,
    eta_f_ra = np.sqrt(0.45), eta_f_dec = np.sqrt(0.45),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = 1.,
    T_rec_raw = 50.,
    t_obs_days = 5 * 365.25,
)

HIRAX1024 = Telescope(
    name = 'HIRAX1024',
    nu_min = 400, nu_max = 800,
    dec = np.deg2rad(-30.7),
    pointing = True,
    survey_ra_max = 2*np.pi, survey_ra_min = 0.,
    fixed_survey_za_max = np.deg2rad(30.),
    size_ra = 32 * 7 * 100, size_dec = 32 * 9 * 100,
    eta_f_ra = np.sqrt(0.45), eta_f_dec = np.sqrt(0.45),
    eta_a_ra = np.sqrt(0.5), eta_a_dec = np.sqrt(0.5),
    primary_beam_baseline_ra = 6 * 100, primary_beam_baseline_dec = 6 * 100,
    eta_sig = 1.,
    T_rec_raw = 50.,
    t_obs_days = 5 * 365.25,
)
