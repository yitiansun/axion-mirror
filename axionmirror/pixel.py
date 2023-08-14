"""Class for pixel grids in one sensitivity scan."""

from functools import partial

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def grid_edges(pixel_size_func, a_min, a_max, a_mid):
    """Pixelization grid edges consistent with a pixel size function."""
    grid_edges = [a_mid]
    a = a_mid
    a += pixel_size_func(a)
    while a < a_max:
        grid_edges.append(a)
        a += pixel_size_func(a)
    grid_edges.append(a_max)
    
    a = a_mid
    a -= pixel_size_func(a)
    while a > a_min:
        grid_edges.insert(0, a)
        a -= pixel_size_func(a)
    grid_edges.insert(0, a_min)
    
    return grid_edges


class PixelConfig:
    """
    Pixel grid configurations for a given telescope, handling the spatial and frequency dimensions.

    Args:
        telescope (Telescope)
        n_nu (int): The number of frequency bins.
        n_ra_shift (int): The number of right ascension shifts.
        n_dec_shift (int): The number of declination shifts.
    """
    def __init__(self, telescope, n_nu, n_ra_shift, n_dec_shift):
        
        self.telescope = telescope
        self.n_nu = n_nu
        self.nu_s = np.linspace(telescope.nu_min, telescope.nu_max, n_nu)

        ra_mid_base = np.pi
        self.n_ra_shift = n_ra_shift
        self.ra_mids = ra_mid_base + np.linspace(0, telescope.ra_pixel_size(ra_mid_base), n_ra_shift, endpoint=False)
        dec_mid_base = telescope.dec
        self.n_dec_shift = n_dec_shift
        self.dec_mids = dec_mid_base + np.linspace(0, telescope.dec_pixel_size(dec_mid_base), n_dec_shift, endpoint=False)


    def build(self, i_nu, i_ra_shift, i_dec_shift):

        self.postfix = f'inu{i_nu}-ira{i_ra_shift}-idec{i_dec_shift}'
        self.nu = self.nu_s[i_nu]
        self.ra_mid = self.ra_mids[i_ra_shift]
        self.dec_mid = self.dec_mids[i_dec_shift]
        self.ra_edges = grid_edges(
            partial(self.telescope.ra_pixel_size, nu=self.nu),
            self.telescope.survey_ra_min,
            self.telescope.survey_ra_max,
            self.ra_mid
        )
        self.dec_edges = grid_edges(
            partial(self.telescope.dec_pixel_size, nu=self.nu),
            self.telescope.survey_dec_min,
            self.telescope.survey_dec_max,
            self.dec_mid
        )
        self.ra_s = (self.ra_edges[1:] + self.ra_edges[:-1]) / 2
        self.dec_s = (self.dec_edges[1:] + self.dec_edges[:-1]) / 2
        self.ra_grid, self.dec_grid = np.meshgrid(self.ra_s, self.dec_s)
        self.radec_shape = (len(self.dec_s), len(self.ra_s))

        coord_grid = SkyCoord(ra=self.ra_grid*u.rad, dec=self.dec_grid*u.rad, frame='icrs')
        self.l_grid = np.array(coord_grid.galactic.l.rad)
        self.b_grid = np.array(coord_grid.galactic.b.rad)

    @property
    def radec_flat(self):
        return np.stack([self.ra_grid.ravel(), self.dec_grid.ravel()], axis=-1)
    
    @property
    def lb_flat(self):
        return np.stack([self.l_grid.ravel(), self.b_grid.ravel()], axis=-1)