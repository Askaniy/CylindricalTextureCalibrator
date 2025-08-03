from typing import Callable
from traceback import format_exc
from time import monotonic
from io import BytesIO
import numpy as np
from math import ceil, sqrt
from scipy.interpolate import interp1d
from PIL import Image


def to_supported_mode(mode: str):
    """ Corresponds the image mode of the Pillow library and supported one """
    # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    match mode:
        case 'P' | 'RGB' | 'RGBX' | 'CMYK' | 'YCbCr' | 'LAB' | 'HSV':
            # 8-bit color
            return 'RGB'
        case 'PA' | 'RGBA' | 'RGBa':
            # 8-bit color with alpha channel
            return 'RGBA'
        case 'L':
            # 8-bit grayscale
            return 'L'
        case 'La' | 'LA':
            # 8-bit grayscale with alpha channel
            return 'LA'
        case 'I' | 'I;16' | 'I;16L' | 'I;16B' | 'I;16N':
            # 32-bit grayscale
            return 'I'
        case 'F':
            # 32-bit floating point grayscale
            return 'F'
        case _:
            print(f'Mode {mode} is not recognized. Would be processed as RGB image.')
            return 'RGB'

def color_depth(mode: str):
    """ Corresponds the image mode of the Pillow library and its bitness """
    match mode:
        case 'RGB' | 'RGBA' | 'L' | 'LA': # 8 bit
            return 255
        case 'I' | 'F': # 32 bit
            return 65535
        case _:
            print(f'Mode {mode} is not supported. Would be processed as 8-bit image.')
            return 255

def img2array(img: Image.Image) -> np.ndarray:
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

def image_reader(file: str, preview_area: int) -> tuple[Image.Image, Image.Image]:
    """ Imports spectral data from a RGB image """
    img = Image.open(file)
    img = img.convert(to_supported_mode(img.mode))
    factor = ceil(sqrt(img.width * img.height / preview_area))
    preview = img.resize((img.width // factor, img.height // factor), Image.Resampling.NEAREST)
    return img, preview

def img2bytes(img: Image.Image) -> bytes:
    """ Prepares PIL's image to be displayed in the window """
    bio = BytesIO()
    img.save(bio, format='png')
    del img
    return bio.getvalue()

def array2img(array: np.ndarray) -> Image.Image:
    """ Creates a Pillow image from the NumPy array """
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype('uint8').transpose())

def color_parser(color: str) -> np.ndarray | None:
    """ Ensures adequate processing of user input of space-separated color """
    try:
        return np.array(color.split(), dtype='float')
    except Exception:
        return None

def float_parser(string: str) -> float | None:
    """ Ensures adequate processing of user input of float value """
    try:
        return eval(string, None)
    except Exception:
        return None

def gamma_correction(arr0: np.ndarray) -> np.ndarray:
    """ Applies gamma correction in CIE sRGB implementation to the array """
    arr1 = np.copy(arr0)
    mask = arr0 < 0.0031308
    arr1[mask] *= 12.92
    arr1[~mask] = 1.055 * np.power(arr1[~mask], 1./2.4) - 0.055
    return arr1

def latitudes(y_len: int):
    """ Returns array of latitudes in radians """
    return ((np.arange(y_len) + 0.5) / y_len - 0.5) * np.pi
    #        centering pixels ^^^^^

def extend(arr: np.ndarray, times: int):
    """ Adds a new zero axis to the array and repeats along it the specified number of times """
    return np.repeat(np.expand_dims(arr, axis=0), times, axis=0)

def map_weights(shape: tuple, obl: float = 0.):
    """
    Returns an area contribution map for a planetographic projection of an oblate spheroid.
    The last two values of `shape` should be X and Y lengths.
    """
    try:
        if obl == 1:
            # An outgrown case
            area = np.zeros(shape[-1])
            area[0] = 1.
            area[-1] = 1.
        else:
            phi = latitudes(shape[-1])
            # Eccentricity squared
            e2 = obl * (2 - obl)
            # Area weights based on planetographic oblate spheroid metric tensor
            area = (1 - e2) * np.cos(phi) / (1 - e2 * np.sin(phi)**2)**2
    except Exception:
        area = np.ones(shape[1])
    return extend(area, shape[-2]) # along X axis

def separate_alpha(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Separates image and alpha channel, handling grayscale and color images.
    Returns None in place of the alpha channel if there is none.
    """
    alpha = None
    if arr.shape[0] == 2:
        # grayscale image
        alpha = arr[1]
        arr = arr[0]
    elif arr.shape[0] == 4:
        # color image
        alpha = arr[3]
        arr = arr[:3]
    return arr, alpha

def color_calibrator(arr: np.ndarray, color: np.ndarray, obl: float = 0):
    """
    Scales the channels so that the average brightnesses match the given color.
    The alpha channel is read as a mask.
    """
    # Alpha channel separation
    arr, alpha = separate_alpha(arr)
    # Calculating cylindrical map weights
    weights = map_weights(arr.shape, obl)
    if alpha is not None:
        weights *= alpha
    # Calculating weighted average and scaling
    if arr.ndim == 2:
        # if grayscale image
        average = np.sum(arr * weights) / weights.sum()
        arr = extend(arr / average, 3)
    else:
        # if color image
        average = np.sum(arr * weights[np.newaxis, ...], axis=(1, 2)) / weights.sum()
        arr /= average.reshape((3, 1, 1))
    # Recoloring the image
    arr *= color.reshape((3, 1, 1))
    # Returning alpha channel
    if alpha is not None:
        arr = np.vstack([arr, alpha[None]])
    return arr

def albedo_calibrator(arr: np.ndarray, albedo: float, obl: float = 0) -> np.ndarray:
    """
    Scales the channels so that the green channel brightness corresponds to the albedo.
    The alpha channel is read as a mask.
    """
    # Alpha channel separation
    arr, alpha = separate_alpha(arr)
    # Calculating cylindrical map weights
    weights = map_weights(arr.shape, obl)
    if alpha is not None:
        weights *= alpha
    # Calculating weighted average
    reference_channel = arr if arr.ndim == 2 else arr[1]
    average = np.sum(reference_channel * weights) / weights.sum()
    # Rescaling the image
    arr = arr / average * albedo
    # Returning alpha channel
    if alpha is not None:
        arr = np.vstack([arr, alpha[None]])
    return arr

def generate_grid_layer(shape: tuple, divisions: int) -> np.ndarray:
    """
    Draws the specified number of lines along the X-axis (and half as many along the Y-axis),
    distributing the brightness to the pixels closest to the true line position
    """
    grid = np.zeros(shape)
    x_step = shape[0] / divisions
    # Drawing lines on X axis
    for i in range(divisions):
        subpixel_num = x_step * i - 0.5
        upper_num = ceil(subpixel_num)
        proportion = upper_num - subpixel_num
        grid[upper_num] = 1-proportion
        grid[upper_num-1] = proportion
    # Drawing lines on Y axis
    y_step = shape[1] / divisions * 2
    for j in range(divisions//2):
        subpixel_num = y_step * j - 0.5
        upper_num = ceil(subpixel_num)
        proportion = upper_num - subpixel_num
        grid[:, upper_num] = 1-proportion
        grid[:, upper_num-1] = proportion
    return grid

def generate_grid(shape: tuple) -> np.ndarray:
    """ Draws different colors of 90°, 30° and 15° degree grids """
    scale90deg = extend(generate_grid_layer(shape, 4), 3)  * np.reshape([1, 1, 1], (3, 1, 1))
    scale30deg = extend(generate_grid_layer(shape, 8), 3)  * np.reshape([1, 1, 0], (3, 1, 1))
    scale15deg = extend(generate_grid_layer(shape, 16), 3) * np.reshape([1, 0, 0], (3, 1, 1))
    return scale15deg + scale30deg + scale90deg

def add_grid(arr: np.ndarray) -> np.ndarray:
    """ Adds a coordinate grid to the image array """
    # Alpha channel separation
    arr, alpha = separate_alpha(arr)
    if arr.ndim == 2:
        # if grayscale image
        arr = extend(arr, 3)
    # Adding the grid
    grid = generate_grid(arr[0].shape)
    mask = grid.max(axis=0)
    arr = arr * (1 - mask) + grid * mask
    # Returning alpha channel
    if alpha is not None:
        alpha = np.clip(alpha + mask, 0, 1)
        arr = np.vstack([arr, alpha[None]])
    return arr

# Exponent constant
_imag_2pi = -1j*2*np.pi

def subpixel_shift(arr: np.ndarray, shift: float):
    """ Subpixel longitude shift, uses the Fast Fourier Transform """
    shift /= 360 # degrees to fraction
    # -2 is longitude axis index place (1 for color mode and 0 for grayscale)
    x_len = arr.shape[-2]
    freq = np.fft.fftfreq(x_len)[:, np.newaxis] * shift * x_len
    kernel = np.exp(_imag_2pi * freq)
    arr = np.nan_to_num(arr)
    return np.real(np.fft.ifftn(np.fft.fftn(arr, axes=(-2,)) * kernel, axes=(-2,)))

def planetocentric2planetographic(arr0: np.ndarray, obl: float = 0.):
    """ Reprojects the map from planetocentric to planetographic latitude system """
    phi0 = latitudes(arr0.shape[2])
    phi1 = np.arctan(np.tan(phi0) * (1-obl)**2)
    arr1 = interp1d(phi0, arr0, kind='cubic', fill_value='extrapolate')(phi1)
    return arr1

def equal_area2planetographic(arr0: np.ndarray, obl: float = 0.):
    """ Reprojects the map from equal-area (Lambert) to planetographic latitude system """
    phi0 = latitudes(ceil(arr0.shape[2]))
    phi1 = latitudes(ceil(arr0.shape[2] * 0.5 * np.pi))
    phi1 = np.sin(phi1) * 0.5 * np.pi
    if obl != 0.:
        phi1 = np.arctan(np.tan(phi1) * (1-obl)**2)
    arr1 = interp1d(phi0, arr0, kind='cubic', fill_value='extrapolate')(phi1)
    return arr1

projections_dict = {
    'planetocentric': planetocentric2planetographic,
    'equal-area': equal_area2planetographic,
}

def image_parser(
        img: Image.Image,
        preview_flag: bool = False,
        save_file: str = '',
        projection: str = '',
        shift: float = None,
        oblateness: float = None,
        albedo_target: float = None,
        color_target: np.ndarray = None,
        sRGB_gamma: bool = False,
        custom_gamma: float = None,
        maximize_brightness: bool = False,
        grid: bool = False,
        log: Callable = print
    ):
    """ Receives user input and performs processing in a parallel thread """
    if not preview_flag:
        start_time = monotonic()
    if oblateness is None:
        oblateness = 0.
    try:
        arr = img2array(img)
        if projection != '':
            arr = projections_dict[projection](arr, oblateness)
        if shift is not None:
            arr = subpixel_shift(arr, shift)
        if color_target is not None:
            arr = color_calibrator(arr, color_target, oblateness)
        if albedo_target is not None:
            arr = albedo_calibrator(arr, albedo_target, oblateness)
        if sRGB_gamma:
            arr = gamma_correction(arr)
        if custom_gamma is not None:
            arr **= custom_gamma
        if maximize_brightness and arr.max() != 0:
            arr /= arr.max()
        if grid:
            arr = add_grid(arr)
        img = array2img(arr)
        if preview_flag:
            log('Preview is ready', img)
        else:
            time = monotonic() - start_time
            speed = img.width * img.height / time
            log(f'Processing took {time:.1f} seconds, average speed is {speed:.1f} px/sec')
            try:
                img.save(save_file)
            except ValueError:
                img.save(save_file + '.png')
    except Exception:
        log(f'Image processing failed with {format_exc(limit=0).strip()}')
        print(format_exc())
