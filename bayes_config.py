# Config file
import os
import numpy as np
import torch
import pickle
def load_2_items(storage_materials,storage_depths):

    with open(storage_materials,'rb') as f:
        materials = pickle.load(f)
    with open(storage_depths,'rb') as f:
        depths = pickle.load(f)
    print(materials)
    print(depths)
    return materials, depths
def seed_everything(seed=42):
    """Fix all random seeds to ensure reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
class Config():
    MAE_MODE = 'Non-average'
    MATERIALS_RECOMMENDED = ['Ag', 'Al', 'Al2O3', 'Au', 'C', 'Cr', 'Cu', 'GaAs', 'Ge', 'HfO2', 'MgO', 'Mo', 'Ni', 'Pd', 'PDMS', 'PMMA', 'Pt', 'Si', 'Si3N4', 'SiC', 'SiO2', 'Ti', 'TIN', 'TiO2', 'W', 'ZnS', 'ZrO2']
    ALL_MATERIALS = ['Ag', 'Al', 'Al2O3', 'Au', 'C', 'Cr', 'Cu', 'GaAs', 'Ge', 'HfO2', 'MgO', 'Mo', 'Ni', 'Pd', 'PDMS', 'PMMA', 'Pt', 'Si', 'Si3N4', 'SiC', 'SiO2', 'Ti', 'TIN', 'TiO2', 'W', 'ZnS', 'ZrO2']
    COLOR_TARGET = [-1,-1,-1]
    SPECTRUM_TARGET = [[-1,-1,-1,-1,-1]]
    DEPTH_MAX = 400 # The max depth of each layer of the proposed structure.
    LAYERNUM = 8 # The number of layers of the proposed structure.
    SEARCH_INTERVAL = 20 # Thickness search interval.
    SEARCH_EPOCH = 10000 # Number of iterations.
    INTERVAL_SPECTRUM = 1000 # Number of spectrum calculation points.
    INTERVAL_COLOR = 200 # Number of color calculation points.

    PATH = r'Material_base/simulation'
    TARGET_DIR = ''
    STORAGE = ''
    STORAGE_MATERIALS = ''
    STORAGE_DEPTHS = ''
    PROMPT = 'I want to design an optical structure which meets the following characteristics: high reflectivity in the ultraviolet band (200-400 nm), high transmittance in the visible light band (400-780 nm), and high reflectivity at 780-2500 nm and it is consists of glasses'
def Create_directory(config,algorythm):

    TARGET_DIR = ''
    for every in config.COLOR_TARGET:
        TARGET_DIR = TARGET_DIR + str(every) + ' '
    for each in config.SPECTRUM_TARGET:
        for every in each:
            TARGET_DIR = TARGET_DIR + str(every) + ','
    config.TARGET_DIR = os.path.join('Visualization', TARGET_DIR)
    os.makedirs(config.TARGET_DIR, exist_ok=True)
    # Create the target directory if it doesn't exist

    config.STORAGE_init = f'sqlite:///{config.TARGET_DIR}/stage0.db'
    config.STORAGE = f'sqlite:///{config.TARGET_DIR}/{algorythm}.db'

    config.STORAGE_MATERIALS = os.path.join(f'{config.TARGET_DIR}',algorythm + 'materials.pkl')
    config.STORAGE_DEPTHS = os.path.join(f'{config.TARGET_DIR}',algorythm + 'depths.pkl')




