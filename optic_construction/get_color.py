import numpy as np
from skimage import color as colour
from skimage.color import deltaE_ciede2000
import cv2

def rgb_to_cie_lch(rgb):
    """
    Convert RGB color values to CIE-LCH color values.

    Parameters:
    RGB (numpy. ndarray): An array of shapes (3,) representing RGB color values, typically ranging from 0 to 255.

    return:
    Numpy.ndarray: an array of shapes (3,) representing CIE-LCH color values, namely brightness (L), saturation (C), and hue angle (h).
    """

    rgb_normalized = rgb / 255.0

    # Convert RGB to CIE L*a*b*
    lab = colour.rgb2lab(rgb_normalized)

    l = lab[0]
    a = lab[1]
    b = lab[2]

    c = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a) * 180 / np.pi
    if h < 0:
        h += 360

    cie_lch = np.array([l, c, h])

    return cie_lch
def rgb_to_lab_perfect(rgb):

    bgr = np.array(rgb, dtype=np.float32).reshape(1, 1, 3)[..., ::-1] / 255.0

    # Convert by OpenCV
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0, 0]
def get_color_error(color_list,target_color_list):
    """Accurate calculation of ΔE."""
    lab1 = rgb_to_lab_perfect(color_list).reshape(1, 1, 3)
    lab2 = rgb_to_lab_perfect(target_color_list).reshape(1, 1, 3)
    return deltaE_ciede2000(lab1, lab2)[0, 0]

def get_color_lab_error(rgb1,rgb2):
    rgb1 = np.array([rgb1], dtype=np.uint8)
    lab = colour.rgb2lab(rgb1)
    print(lab)
    rgb2 = np.array([rgb2], dtype=np.uint8)
    lab2 = colour.rgb2lab(rgb2)
    print(lab2)
    squared_error = np.square(lab - lab2)
    print(squared_error)

