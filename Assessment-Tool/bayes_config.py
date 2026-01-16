
import os
import numpy as np
import torch

class Config():
    DEPTH_MAX = 300
    LAYERNUM = 16
    SEARCH_INTERVAL = 20
    SEARCH_EPOCH = 5000
    COLOR_TARGET = [-1, -1, -1]
    SPECTRUM_TARGET = [[400, 525, 1, 1, 'T'], [400, 525, 0, 1, 'R'], [525, 575, 1, 1, 'R'], [525, 575, 0, 1, 'T'], [575, 1100, 0, 1, 'R'], [575, 1100, 1, 1, 'T']]
    KEYWORDS=[]
    Total_count = 0
    Correct_count = 0
    Tool_last_use = 4
