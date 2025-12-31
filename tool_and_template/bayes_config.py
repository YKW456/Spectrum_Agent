# 配置文件,可以直接传递参数到其余py文件
import os
import numpy as np
import torch
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
class Config():
    MATERIALS_RECOMMENDED = ['Ag', 'Ge', 'ZnS', 'ZnO', 'MgO', 'TiN', 'AlN', 'Al', 'Si', 'HfO2', 'Al2O3', 'Ta2O5', 'MgF2', 'SiO2', 'TiO2', 'ITO', 'Si3N4', 'ZnSe']
    DEPTH_MAX = 300
    LAYERNUM = 16
    SEARCH_INTERVAL = 10
    SEARCH_EPOCH = 5000
    PATH = r'Material_base/new'


    COLOR_TARGET = [-1,-1,-1]

    SPECTRUM_TARGET = [[-1,-1,-1,-1,-1]]
    TARGET_DIR = ''
    # Create the target directory if it doesn't exist
    for every in COLOR_TARGET:
        TARGET_DIR = TARGET_DIR + str(every) + ' '
    for each in SPECTRUM_TARGET:
        for every in each:
            TARGET_DIR = TARGET_DIR + str(every) + ','
    TARGET_DIR = os.path.join('Visualization', TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)
    STORAGE = f'sqlite:///{TARGET_DIR}/stage1.db'
    STORAGE_MATERIALS = os.path.join(f'{TARGET_DIR}', 'materials')
    STORAGE_DEPTHS = os.path.join(f'{TARGET_DIR}', 'depths')
    STORAGE_MATERIALS += '.pkl'
    STORAGE_DEPTHS += '.pkl'

