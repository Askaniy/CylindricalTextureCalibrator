from typing import Callable
from traceback import format_exc
from time import strftime, monotonic
from io import BytesIO
import numpy as np
from math import ceil, sqrt
from PIL import Image


def to_supported_mode(mode: str):
    """ Corresponds the image mode of the Pillow library and supported one """
    # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    match mode:
        case 'P' | 'PA' | 'RGB' | 'RGBA' | 'RGBX' | 'RGBa' | 'CMYK' | 'YCbCr' | 'LAB' | 'HSV': # 8-bit indexed color palette, alpha channels, color spaces
            return 'RGB'
        case 'L' | 'La' | 'LA': # 8-bit grayscale
            return 'L'
        case 'I' | 'I;16' | 'I;16L' | 'I;16B' | 'I;16N' | 'BGR;15' | 'BGR;16' | 'BGR;24': # 32-bit grayscale
            return 'I'
        case 'F': # 32-bit floating point grayscale
            return 'F'
        case _:
            print(f'Mode {mode} is not recognized. Would be processed as RGB image.')
            return 'RGB'

def color_depth(mode: str):
    """ Corresponds the image mode of the Pillow library and its bitness """
    match mode:
        case 'RGB' | 'L': # 8 bit
            return 255
        case 'I' | 'F': # 32 bit
            return 65535
        case _:
            print(f'Mode {mode} is not supported. Would be processed as 8-bit image.')
            return 255

def img2array(img: Image.Image):
    """
    Converting a Pillow image to a numpy array
    1.5-2.5 times faster than np.array() and np.asarray()
    Based on https://habr.com/ru/articles/545850/
    """
    img.load()
    e = Image._getencoder(img.mode, 'raw', img.mode)
    e.setimage(img.im)
    shape, type_str = Image._conv_type_shape(img)
    data = np.empty(shape, dtype=np.dtype(type_str))
    mem = data.data.cast('B', (data.data.nbytes,))
    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError(f'encoder error {s} in tobytes')
    return np.transpose(data.astype('float64') / color_depth(img.mode))

def image_reader(file: str, preview_area: tuple) -> np.ndarray:
    """ Imports spectral data from a RGB image """
    img = Image.open(file)
    img = img.convert(to_supported_mode(img.mode))
    factor = ceil(sqrt(img.width * img.height / preview_area))
    preview = img.resize((img.width // factor, img.height // factor), Image.Resampling.NEAREST)
    return img, preview

def convert_to_bytes(img: Image.Image):
    """ Prepares PIL's image to be displayed in the window """
    bio = BytesIO()
    img.save(bio, format='png')
    del img
    return bio.getvalue()

def array2img(array: np.ndarray):
    """ Creates a Pillow image from the NumPy array """
    return Image.fromarray((255 * array).astype('uint8').transpose())

def color_parser(color: str):
    """ Ensures adequate processing of user input of space-separated color """
    try:
        return np.array(color.split(), dtype='float')
    except Exception:
        return None

def float_parser(string: str):
    """ Ensures adequate processing of user input of float value """
    try:
        return eval(string, None)
    except Exception:
        return None

def gamma_correction(arr0: np.ndarray):
    """ Applies gamma correction in CIE sRGB implementation to the array """
    arr1 = np.copy(arr0)
    mask = arr0 < 0.0031308
    arr1[mask] *= 12.92
    arr1[~mask] = 1.055 * np.power(arr1[~mask], 1./2.4) - 0.055
    return arr1

def map_weights(shape: tuple, obl: float = 0):
    """
    Returns an area contribution map for a planetographic cylindrical
    projection of an ellipsoid of rotation.
    `obl` is oblateness of the spheroid.
    """
    return np.ones(shape) # stub

def color_calibrator(arr: np.ndarray, color: np.ndarray):
    """ Scales the channels so that the average brightnesses match the given color """
    weights = np.repeat(np.expand_dims(map_weights(arr[0].shape), axis=0), 3, axis=0)
    means = np.average(arr, weights=weights, axis=(1, 2), keepdims=True)
    return arr / means * color.reshape((3, 1, 1))

def albedo_calibrator(arr: np.ndarray, albedo: float):
    """ Scales the channels so that the green channel brightness corresponds to the albedo """
    green_mean = np.average(arr[1], weights=map_weights(arr[0].shape))
    return arr / green_mean * albedo

def image_parser(
        img: Image,
        preview_flag: bool = False,
        save_folder: str = '',
        albedo_target: float = None,
        color_target: tuple = None,
        sRGB_gamma: bool = False,
        custom_gamma: float = None,
        log: Callable = print
    ):
    """ Receives user input and performs processing in a parallel thread """
    log('Starting the image processing thread')
    start_time = monotonic()
    try:
        arr = img2array(img)
        if color_target is not None:
            arr = color_calibrator(arr, color_target)
        if albedo_target is not None:
            arr = albedo_calibrator(arr, albedo_target)
        if sRGB_gamma:
            arr = gamma_correction(arr)
        if custom_gamma is not None:
            arr = arr**custom_gamma
        img = array2img(arr)
        time = monotonic() - start_time
        speed = img.width * img.height / time
        log(f'Processing took {time:.1f} seconds, average speed is {speed:.1f} px/sec')
        if preview_flag:
            log('Sending the resulting preview to the main thread', img)
        else:
            img.save(f'{save_folder}/CTC_{strftime("%Y-%m-%d_%H-%M-%S")}.png')
    except Exception:
        log(f'Image processing failed with {format_exc(limit=0).strip()}')
        print(format_exc())