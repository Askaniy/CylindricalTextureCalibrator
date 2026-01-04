from copy import deepcopy
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
            # 8-bit int color
            return 'RGB'
        case 'PA' | 'RGBA' | 'RGBa':
            # 8-bit int color with alpha channel
            return 'RGBA'
        case 'L':
            # 8-bit int grayscale
            return 'L'
        case 'La' | 'LA':
            # 8-bit int grayscale with alpha channel
            return 'LA'
        case 'I' | 'I;16' | 'I;16L' | 'I;16B' | 'I;16N':
            # 32-bit int grayscale
            return 'I'
        case 'F':
            # 32-bit float grayscale
            return 'F'
        case _:
            print(f'Mode {mode} is not recognized. Would be processed as RGB image.')
            return 'RGB'

def color_depth(mode: str):
    """ Corresponds the image mode of the Pillow library and its bitness """
    match mode:
        case 'RGB' | 'RGBA' | 'L' | 'LA':
            # 8-bit int
            return 255
        case 'I':
            # 32-bit int
            return 65535
        case 'F':
            # 32-bit float
            return 1
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

def color_parser(color: str) -> np.ndarray | None:
    """ Ensures adequate processing of user input of space-separated color """
    try:
        return np.array(color.split(), dtype='float')
    except Exception:
        return None

def float_parser(string: str) -> float:
    """ Ensures adequate processing of user input of float value """
    try:
        return float(eval(string, None))
    except Exception:
        return 0.

def latitudes(y_len: int):
    """ Returns array of latitudes in radians """
    return ((np.arange(y_len) + 0.5) / y_len - 0.5) * np.pi
    #        centering pixels ^^^^^

def extend(arr: np.ndarray, times: int):
    """ Adds a new zero axis to the array and repeats along it the specified number of times """
    return np.repeat(np.expand_dims(arr, axis=0), times, axis=0)

def map_weights(width: int, height: int, obl: float = 0.):
    """
    Returns an area contribution map for a planetographic latitude system
    of an oblate spheroid.
    """
    try:
        if obl == 1:
            # An outgrown case
            area = np.zeros(height)
            area[0] = 1.
            area[-1] = 1.
        else:
            phi = latitudes(height)
            # Eccentricity squared
            e2 = obl * (2 - obl)
            # Area weights based on planetographic oblate spheroid metric tensor
            area = (1 - e2) * np.cos(phi) / (1 - e2 * np.sin(phi)**2)**2
    except Exception:
        area = np.ones(height)
    result = extend(area, width) # along X axis
    return result

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


def planetographic2planetographic(arr0: np.ndarray, _):
    """ Does nothing """
    return arr0

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

latitude_systems_dict = {
    'planetographic': planetographic2planetographic,
    'planetocentric': planetocentric2planetographic,
    'equal-area': equal_area2planetographic,
}


class ImageArray:
    """
    Class that operates with an array of image values,
    its alpha channel, and other image proprieties.

    Attributes:
    - values
    - alpha
    - is_grayscale
    - is_cylindrical_map
    - oblateness
    - width
    - height
    - size
    """

    def __init__(
            self,
            img: Image.Image,
            is_gamma_corrected: bool,
            is_cylindrical_map: bool,
            latitude_system: str,
            oblateness: int|float
        ) -> None:
        """
        Creates an ImageArray from the Pillow image
        and applies preprocessing steps
        """
        self.values = img2array(img)
        # Separating values and alpha, if needed
        if len(self.values.shape) == 2:
            # L mode
            self.alpha = None
        else:
            match self.values.shape[0]:
                case 2: # LA mode
                    self.alpha = self.values[1]
                    self.values = self.values[0]
                case 3: # RGB mode
                    self.alpha = None
                case 4: # RGBA mode
                    self.alpha = self.values[3]
                    self.values = self.values[:3]
                case _:
                    raise ValueError(f'ImageArray initialization error: {self.values.shape[0]} is not a valid number of channels')
        self.is_cylindrical_map = is_cylindrical_map
        self.oblateness = oblateness
        self.is_grayscale = self.values.ndim == 2
        # Linearization if the original image is already gamma corrected
        if is_gamma_corrected:
            self.undo_gamma_correction()
        # Reprojection if the original image's latitude system is not planetographic
        self.values = latitude_systems_dict[latitude_system](self.values, oblateness)

    def get_array(self):
        """ Returns combined arrays of brightness values and alpha channel """
        if self.alpha is None:
            arr = self.values
        else:
            arr = np.vstack((self.values, self.alpha[None]))
        return arr

    def to_image(self):
        """ Creates a Pillow image from the ImageArray """
        # TODO: save in 16 bit? edit meta data, set gamma correction?
        return Image.fromarray(np.round(np.clip(self.get_array(), 0, 1) * 255).astype('uint8').transpose())

    def to_bytes(self):
        """ Prepares ImageArray to be displayed in the window """
        return img2bytes(self.to_image())

    def weights(self):
        """
        Calculates 2D array of brightness contribution weights,
        accounting image proprieties and the alpha channel.
        """
        if self.is_cylindrical_map:
            # latitude-dependent distribution (planetographic)
            weights = map_weights(self.width, self.height, self.oblateness)
        else:
            # uniform distribution
            weights = np.ones((self.width, self.height))
        if self.alpha is not None:
            weights *= self.alpha
        return weights

    def mean_brightness(self):
        # Calculating image brightness contribution weights
        weights = self.weights()
        # Calculating weighted mean and scaling
        if self.is_grayscale:
            total_value = np.sum(self.values * weights)
        else:
            total_value = np.sum(self.values * weights[np.newaxis, ...], axis=(1, 2))
        return total_value / weights.sum()

    def formatted_mean_brightness(self) -> str:
        if self.is_grayscale:
            text = f'Mean brightness: {self.mean_brightness():.5f}'
        else:
            r, g, b = self.mean_brightness()
            text = f'Mean RGB: ({r:.5f}, {g:.5f}, {b:.5f})'
        return text

    def calibrate_color(self, color_target: np.ndarray):
        """
        Creates a new ImageArray object with scaled channels
        so that the mean brightnesses match the given color.
        """
        other = deepcopy(self)
        if self.is_grayscale:
            arr = extend(self.values / self.mean_brightness(), 3) # grayscale to RGB
            other.is_grayscale = False
        else:
            arr = self.values / self.mean_brightness().reshape((3, 1, 1)) # to match the shape
        other.values = arr * color_target.reshape((3, 1, 1)) # to match the shape
        return other

    def calibrate_albedo(self, albedo_target: float):
        """
        Creates a new ImageArray object with scaled channels
        so that the green channel brightness corresponds to the albedo.
        """
        # Calculating image brightness contribution weights
        weights = self.weights()
        # Calculating weighted mean and scaling
        if self.is_grayscale:
            reference_channel = self.values
        else:
            reference_channel = self.values[1] # green
        mean_value = np.sum(reference_channel * weights) / weights.sum()
        other = deepcopy(self)
        other.values = self.values / mean_value * albedo_target
        return other

    def apply_gamma_correction(self):
        """
        Creates a new ImageArray object with applied gamma correction
        in CIE sRGB implementation.
        """
        mask = self.values < 0.0031308
        other = deepcopy(self)
        other.values[mask] *= 12.92
        other.values[~mask] = 1.055 * np.power(self.values[~mask], 1./2.4) - 0.055
        return other

    def undo_gamma_correction(self):
        """ Applies inverse gamma correction in CIE sRGB implementation """
        mask = np.copy(self.values) <= 0.04045
        self.values[mask] /= 12.92
        self.values[~mask] = np.power((self.values[~mask] + 0.055) / 1.055, 2.4)

    def maximize_brightness(self):
        """ Creates a new ImageArray object with brightness being maximized """
        max_value = self.values.max()
        if max_value != 0:
            other = deepcopy(self)
            other.values /= max_value
            return other
        else:
            return self

    # Exponent constant
    __imag_2pi = -1j*2*np.pi

    def subpixel_shift(self, shift: float):
        """
        Creates a new ImageArray object with applied subpixel longitude shift
        in degrees, using the Fast Fourier Transform.
        """
        shift /= 360 # degrees to fraction
        freq = np.fft.fftfreq(self.width)[:, np.newaxis] * shift * self.width
        kernel = np.exp(self.__imag_2pi * freq)
        other = deepcopy(self)
        other.values = np.real(np.fft.ifftn(np.fft.fftn(self.values, axes=(-2,)) * kernel, axes=(-2,)))
        if self.alpha is not None:
            other.alpha = np.real(np.fft.ifftn(np.fft.fftn(self.alpha, axes=(-2,)) * kernel, axes=(-2,)))
        return other

    def add_grid(self):
        """ Creates a new ImageArray object with added coordinate grid """
        other = deepcopy(self)
        if self.is_grayscale:
            other.values = extend(self.values, 3) # grayscale to RGB
            other.is_grayscale = False
        # Adding the grid
        grid = generate_grid(other.values[0].shape)
        mask = grid.max(axis=0)
        other.values = self.values * (1 - mask) + grid * mask
        # Removing lines transparency
        if self.alpha is not None:
            other.alpha = np.clip(self.alpha + mask, 0, 1)
        return other

    @property
    def width(self):
        """ Returns horizontal spatial axis length """
        if self.is_grayscale:
            return self.values.shape[0]
        else:
            return self.values.shape[1]

    @property
    def height(self):
        """ Returns vertical spatial axis length """
        if self.is_grayscale:
            return self.values.shape[1]
        else:
            return self.values.shape[2]

    @property
    def size(self):
        """ Returns the number of pixels """
        return self.width * self.height




def image_parser(
        img_arr: ImageArray,
        preview_flag: bool = False,
        save_file: str = '',
        shift: int|float = None,
        albedo_target: float = None,
        color_target: np.ndarray = None,
        sRGB_gamma: bool = False,
        maximize_brightness: bool = False,
        grid: bool = False,
        log: Callable = print
    ):
    """ Receives user input and performs processing in a parallel thread """
    if not preview_flag:
        start_time = monotonic()
    try:
        if shift is not None and shift != 0:
            img_arr = img_arr.subpixel_shift(shift)
        if color_target is not None:
            img_arr = img_arr.calibrate_color(color_target)
        if albedo_target is not None:
            img_arr = img_arr.calibrate_albedo(albedo_target)
        if sRGB_gamma:
            img_arr = img_arr.apply_gamma_correction()
        if maximize_brightness:
            img_arr = img_arr.maximize_brightness()
        if grid:
            img_arr = img_arr.add_grid()
        if preview_flag:
            log('Preview is ready', img_arr)
        else:
            img = img_arr.to_image()
            time = monotonic() - start_time
            speed = img_arr.width * img_arr.height / time
            log(f'Processing took {time:.1f} seconds, average speed is {speed:.1f} px/sec')
            try:
                img.save(save_file)
            except ValueError:
                img.save(save_file + '.png')
    except Exception:
        log(f'Image processing failed with {format_exc(limit=0).strip()}')
        print(format_exc())
