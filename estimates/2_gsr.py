import sys
sys.path.append("..")
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

from tqdm import tqdm
import pickle
import h5py

import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import jax.numpy as jnp

from config import config_dict, intermediates_dir
from aatw.units_constants import *
from aatw.nfw import rho_integral, rho_integral_ref
from aatw.spectral import dnu, prefac
from aatw.map_utils import pad_mbl, interp2d_vmap


def gsr(run_dir, remove_GCantiGC=True, field_model=..., telescope=..., nu_arr=...,
        i_nu=..., i_ra_grid_shift=..., i_dec_grid_shift=..., **kwargs):
    """Make signal and background maps for Galactic Synchrotron Radiation (GSR)"""
    
    nu = nu_arr[i_nu]
    
    subrun_postfix = f'inu{i_nu}-ira{i_ra_grid_shift}-idec{i_dec_grid_shift}'
    
    coords_dict = pickle.load(open(f'{run_dir}/coords/coords-{subrun_postfix}.p', 'rb'))
    ra_s = coords_dict['ra_s']
    dec_s = coords_dict['dec_s']
    ra_edges = coords_dict['ra_edges']
    dec_edges = coords_dict['dec_edges']
    ra_grid = coords_dict['ra_grid']
    dec_grid = coords_dict['dec_grid']
    radec_flat = coords_dict['radec_flat']
    radec_shape = coords_dict['radec_shape']
    l_grid = coords_dict['l_grid']
    b_grid = coords_dict['b_grid']
    
    #========== Source (Haslam) map ==========
    nu_haslam = 408 # [MHz]
    beta = -2.5
    # destriped (not desourced) map for background
    haslam_ds_map_hp = hp.read_map('../data/gsr/haslam408_ds_Remazeilles2014.fits')
    haslam_ds_map_hp *= (nu/nu_haslam) ** beta
    haslam_ds_map = hp.pixelfunc.get_interp_val(
        haslam_ds_map_hp, np.rad2deg(l_grid), np.rad2deg(b_grid), lonlat=True
    )
    
    #========== Signal ==========
    l_grid = jnp.where(l_grid > np.pi, l_grid - 2*np.pi, l_grid)
    bl_flat = jnp.stack([b_grid.ravel(), l_grid.ravel()], axis=-1)
    
    if field_model != 'JF':
        raise NotImplementedError(field_model)
    
    with h5py.File(f"../outputs/gsr/Ta_408MHz_field{field_model}.h5") as hf: # Ta: signal temperature for all
        padded_Ta, padded_b, padded_l = pad_mbl(hf['Ta'][:], hf['b_s'][:], hf['l_s'][:])
    
    sig_temp_map = interp2d_vmap(
        jnp.asarray(padded_Ta),
        jnp.asarray(padded_b),
        jnp.asarray(padded_l),
        bl_flat
    ).reshape(radec_shape)
    
    sig_temp_map = np.array(sig_temp_map) * (nu/nu_haslam)**beta * (prefac(nu)/prefac(nu_haslam))
    
    #----- remove GC pixel and anti-GC pixel -----
    if remove_GCantiGC:
        coord_GC = SkyCoord(l=0*u.rad, b=0*u.rad, frame='galactic')
        ra_GC = coord_GC.icrs.ra.rad
        dec_GC = coord_GC.icrs.dec.rad

        i_ra_GC = np.searchsorted(ra_edges, ra_GC) - 1 # v_edges[i-1] < v < v_edges[i]
        i_dec_GC = np.searchsorted(dec_edges, dec_GC) - 1
        if 0 < i_dec_GC and i_dec_GC < len(dec_s): # checking only DEC because RA must be in range
            sig_temp_map[i_dec_GC, i_ra_GC] = 0

        ra_antiGC = ra_GC+np.pi if ra_GC < np.pi else ra_GC-np.pi
        dec_antiGC = - dec_GC

        i_ra_antiGC = np.searchsorted(ra_edges, ra_antiGC) - 1 # v_edges[i-1] < v < v_edges[i]
        i_dec_antiGC = np.searchsorted(dec_edges, dec_antiGC) - 1
        if 0 < i_dec_antiGC and i_dec_antiGC < len(dec_s): # checking only DEC because RA must be in range
            sig_temp_map[i_dec_antiGC, i_ra_antiGC] = 0

    #========== Exposure ==========
    ra_beam_size = (c0 / nu) / telescope.primary_beam_baseline
    sec_in_day = 86400 # [s]
    exposure_map = sec_in_day / (2*np.pi / ra_beam_size)
    if telescope.double_pass_dec is not None:
        exposure_map *= ((dec_grid > telescope.double_pass_dec) + 1)

    #========== Background ==========
    T_sys = telescope.T_sys(nu) if callable(telescope.T_sys) else telescope.T_sys
    eta   = telescope.eta(nu)   if callable(telescope.eta)   else telescope.eta
    bkg_temp_map = haslam_ds_map + T_sys / eta # [K/eta]

    #========== Save ==========
    os.makedirs(f"{run_dir}/gsr_{field_model}", exist_ok=True)
    os.makedirs(f"{run_dir}/bkg", exist_ok=True)
    os.makedirs(f"{run_dir}/exposure", exist_ok=True)
    np.save(f'{run_dir}/gsr_{field_model}/gsr-{subrun_postfix}.npy', sig_temp_map)
    np.save(f'{run_dir}/bkg/bkg-{subrun_postfix}.npy', bkg_temp_map)
    np.save(f'{run_dir}/exposure/exposure-{subrun_postfix}.npy', exposure_map)
    

if __name__ == "__main__":
    
    config_name = 'HIRAX-1024-nnu30-nra3-ndec3'
    config = config_dict[config_name]
    
    for i_nu in tqdm(range(len(config['nu_arr']))):
        for i_ra in range(config['n_ra_grid_shift']):
            for i_dec in range(config['n_dec_grid_shift']):
                gsr(
                    run_dir=f'{intermediates_dir}/{config_name}', field_model='JF', remove_GCantiGC=True,
                    i_nu=i_nu, i_ra_grid_shift=i_ra, i_dec_grid_shift=i_dec, **config
                )
