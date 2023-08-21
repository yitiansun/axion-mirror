import sys

import numpy as np

sys.path.append("..")
from axionmirror.telescopes import CHIME, HERA, CHORD, HIRAX256, HIRAX1024, BURSTT256, BURSTT2048
from axionmirror.pixel import PixelConfig


wdir = '/n/holyscratch01/iaifi_lab/yitians/all_sky_gegenschein/axion-mirror/outputs/intermediates'

pc_dict = {}
pc_dict['CHIME'] = PixelConfig(CHIME, 10, 1, 1, wdir)
pc_dict['HERA']  = PixelConfig(HERA, 10, 1, 8, wdir)
pc_dict['CHORD'] = PixelConfig(CHORD, 10, 1, 1, wdir)
pc_dict['HIRAX256'] = PixelConfig(HIRAX256, 10, 1, 1, wdir)
pc_dict['HIRAX1024'] = PixelConfig(HIRAX1024, 10, 1, 1, wdir)
pc_dict['BURSTT256'] = PixelConfig(BURSTT256, 10, 1, 1, wdir)
pc_dict['BURSTT2048'] = PixelConfig(BURSTT2048, 10, 1, 1, wdir)