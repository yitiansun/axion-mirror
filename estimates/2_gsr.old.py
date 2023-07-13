import os
import sys
from tqdm import tqdm
import pickle
import h5py

import numpy as np
import healpy as hp
from astropy.io import fits
import jax.numpy as jnp

from config import config_dict, intermediates_dir

sys.path.append("..")
from aatw.units_constants import *
from aatw.nfw import rho_integral, rho_integral_ref
from aatw.spectral import dnu, prefac
from aatw.map_utils import interpolate_padded

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


def gsr(run_dir, naive_gegenschein=False, field_model=...,
        telescope=..., nu_arr=..., i_nu=..., i_ra_grid_shift=..., i_dec_grid_shift=..., **kwargs):
    """Make signal and background maps for Galactic Synchrotron Radiation (GSR)"""
    
    nu = nu_arr[i_nu]
    
    subrun_postfix = f'inu{i_nu}-ira{i_ra_grid_shift}-idec{i_dec_grid_shift}'
    
    coords_dict = pickle.load(open(f'{run_dir}/coords/coords-{subrun_postfix}.p', 'rb'))
    ra_s = coords_dict['ra_s']
    dec_s = coords_dict['dec_s']
    ra_grid = coords_dict['ra_grid']
    dec_grid = coords_dict['dec_grid']
    radec_flat = coords_dict['radec_flat']
    radec_shape = coords_dict['radec_shape']
    l_grid = coords_dict['l_grid']
    b_grid = coords_dict['b_grid']
    lb_flat = jnp.asarray(coords_dict['lb_flat'])
    
    anti_l_grid = l_grid + np.pi
    anti_l_grid = np.where(anti_l_grid > 2*np.pi, anti_l_grid - 2*np.pi, anti_l_grid)
    anti_b_grid = - b_grid
    anti_lb_flat = np.stack([anti_l_grid.ravel(), anti_b_grid.ravel()], axis=-1)
    
    #========== Source (Haslam) map ==========
    nu_haslam = 408 # [MHz]
    beta = -2.55
    # destriped desourced
    haslam_dsds_map_hp = hp.read_map('../data/gsr/haslam408_dsds_Remazeilles2014.fits')
    haslam_dsds_map_hp *= (nu/nu_haslam) ** beta
    haslam_dsds_map = hp.pixelfunc.get_interp_val(
        haslam_dsds_map_hp, np.rad2deg(l_grid), np.rad2deg(b_grid), lonlat=True
    )
    # destriped
    haslam_ds_map_hp = hp.read_map('../data/gsr/haslam408_ds_Remazeilles2014.fits')
    haslam_ds_map_hp *= (nu/nu_haslam) ** beta
    haslam_ds_map = hp.pixelfunc.get_interp_val(
        haslam_ds_map_hp, np.rad2deg(l_grid), np.rad2deg(b_grid), lonlat=True
    )

    #========== DM column integral ==========
    rho_integral_map = rho_integral(lb_flat).reshape(radec_shape)

    #========== Naive gegenschein / reference forwardschein temperature ==========
    haslam_anti_dsds_map = hp.pixelfunc.get_interp_val(
        haslam_dsds_map_hp, anti_l_grid/deg, anti_b_grid/deg, lonlat=True
    )
    gegen_temp_map = prefac(nu) * haslam_anti_dsds_map * rho_integral_map # naive gegenschein
    forward_temp_map = prefac(nu) * haslam_dsds_map * rho_integral_ref
    # reference forward+front temperature # I_src * 10kpc uniform column with rho_NFW(r_Sun)
    
    #========== Exposure ==========
    ra_beam_size = (c0 / nu) / telescope.primary_beam_baseline
    sec_in_day = 86400 # [s]
    exposure_map = sec_in_day / (2*np.pi / ra_beam_size)
    if telescope.double_pass_dec is not None:
        exposure_map *= ((dec_grid > telescope.double_pass_dec) + 1)

    #========== Naive S/N ==========
    sig_temp_map = gegen_temp_map
    T_sys = telescope.T_sys(nu) if callable(telescope.T_sys) else telescope.T_sys
    eta   = telescope.eta(nu)   if callable(telescope.eta)   else telescope.eta
    bkg_temp_map = haslam_ds_map + T_sys / eta # [K/eta]
#     SNR_map = sig_temp_map / bkg_temp_map * np.sqrt(
#         2 * dnu(nu) * 1e6 * exposure_map * telescope.t_obs_days
#     )

    #========== S/N with modification according to 3D model ==========
    if not naive_gegenschein:
        #I_data = pickle.load(open(f'../data/galactic_models/I_data_{field_model}.dict', 'rb'))
        
        I_data = {}
        with h5py.File(f'../data/gsr/I_data_{field_model}.h5', 'r') as hf:
            for k, item in hf.items():
                I_data[k] = item[:]
        
        g_ratio_lr = I_data['focused'] / I_data['naive']
        f_ratio_lr = (I_data['front'] + I_data['forward']) / I_data['forward_ref']

        lb_flat_minuspi_to_pi = np.where(lb_flat > np.pi, lb_flat-2*np.pi, lb_flat)
        g_ratio = interpolate_padded(
            g_ratio_lr, I_data['l'], I_data['b'], lb_flat_minuspi_to_pi
        ).reshape(radec_shape)
        f_ratio = interpolate_padded(
            f_ratio_lr, I_data['l'], I_data['b'], lb_flat_minuspi_to_pi
        ).reshape(radec_shape)

        sig_temp_map = gegen_temp_map * g_ratio + forward_temp_map * f_ratio
        bkg_temp_map = haslam_ds_map + T_sys / eta

    np.save(f'{run_dir}/gsr_{field_model}/gsr-{subrun_postfix}.npy', sig_temp_map)
    np.save(f'{run_dir}/bkg/bkg-{subrun_postfix}.npy', bkg_temp_map)
    np.save(f'{run_dir}/exposure/exposure-{subrun_postfix}.npy', exposure_map)
      

if __name__ == "__main__":
    
    config_name = 'HERA-nnu30-nra3-ndec3'
    config = config_dict[config_name]
    
    os.makedirs(f"{intermediates_dir}/{config_name}/gsr_JF", exist_ok=True)
    os.makedirs(f"{intermediates_dir}/{config_name}/bkg", exist_ok=True)
    os.makedirs(f"{intermediates_dir}/{config_name}/exposure", exist_ok=True)
    
    for i_nu in tqdm(range(len(config['nu_arr']))):
        for i_ra in range(config['n_ra_grid_shift']):
            for i_dec in range(config['n_dec_grid_shift']):
                gsr(
                    run_dir=f'{intermediates_dir}/{config_name}',
                    naive_gegenschein=False, field_model='JF',
                    i_nu=i_nu, i_ra_grid_shift=i_ra, i_dec_grid_shift=i_dec, **config
                )
