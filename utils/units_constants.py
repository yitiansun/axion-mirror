import jax.numpy as jnp

##############################
## units: [cm] [MHz] [g] [K] [rad] | SNR: [kpc] [yr]
# constants:
kb = 1.38065e-28 # [cm^2 MHz^2 g / K]
c0 = 29979.2 # [cm MHz]
c0_kpc_yr = 0.000306601 # [kpc/yr]
hbar = 1.05457e-33 # [cm^2 MHz g]

# derived
hour = 3.6e9 # [MHz^-1]
GeV = 1.60218e-15 # [cm^2 MHz^2 g]
invGeV = 1/GeV # [cm^-2 MHz^-2 g^-1]
Jy = 1e-35 # [MHz^2 g]
kpc = 3.08568e21 # [cm]
deg = jnp.pi/180 # [rad]
arcmin = deg/60 # [rad]

# astrophysics
r_Sun = 8.22 # [kpc]
sigmad_over_c = 116/300000 # [1] | (km/s) / (km/s)