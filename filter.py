from skimage import data, io, filters
from skimage.morphology import square, disk, rectangle
from skimage.filters.rank import median, noise_filter, minimum
from math import ceil
import numpy as np

from directory import fruit_basket, image_noise
from error_handeling import KernelShapeError


def read_image(image_array, im_show=False, as_grey=True, flatten=True):
    """read image and show

    Parameters
    ----------
    image_array:numpy.array:
        input image

    Other Parameters
    ----------------
    im_show : Bool
        if True show result
    as_grey: bool
        convert image to grey
    flatten: bool
        convert to 2D

    Returns
    -------
    image : numpy.array
        gray scale image
    """
    image=io.imread(image_array, as_grey=as_grey, flatten=flatten)
    if im_show:
        show_image(image)

    return image

def show_image(image):
    io.imshow(image)
    io.show()

def check_windows_size(windows_size):
    '''
    KernelShapeError if kernel size is Even
    '''
    if windows_size%2==0:
        raise KernelShapeError("windows_size must be odd.")

def make_kernel(window_shape, windows_size, convert_float=False):
    '''
    KernelShapeError if shape is not in ["square", "disk"]
    '''
    if not window_shape in ["square", "disk"]:
        raise KernelShapeError("window_shape is unknown. window_shape={square, disk}.")
    elif window_shape=="square":
        window=square(windows_size)
    elif window_shape=="disk":
        window=disk(windows_size)

    #window convert to float
    if convert_float:
        window=window.astype(float)

    return window


def median_image(image_array, windows_size, window_shape="square", im_show=True):
    """local median of an image.

    Parameters
    ----------
    image_array:numpy.array:
        input image

    windows_size:int
        the size of window

    window_shape: str
        str is element from dict:{square, disk}

    im_show : Bool
        if True show result

    """

    check_windows_size(windows_size)
    kernel=make_kernel(window_shape, windows_size)

    #median
    img=median(image_array, kernel)

    #show image
    if im_show:
        show_image(img)

def noise_filter_image(image_array, windows_size, window_shape="square", im_show=True):
    """Noise feature.

    Parameters
    ----------
    image_array:numpy.array:
        input image

    windows_size:int
        the size of window

    window_shape: str
        str is element from dict:{square, disk}

    im_show : Bool
        if True show result

    """
    check_windows_size(windows_size)
    kernel=make_kernel(window_shape, windows_size)

    #noise
    img=noise_filter(image_array, kernel)

    #show image
    if im_show:
        show_image(img)

def minimum_image(image_array, windows_size, window_shape="square", im_show=True):
    """The lower algorithm complexity makes skimage.filters.rank.minimum more efficient for larger images and structuring elements.

    Parameters
    ----------
    image_array:numpy.array:
        input image

    windows_size:int
        the size of window

    window_shape: str
        str is element from dict:{square, disk}

    im_show : Bool
        if True show result

    """
    check_windows_size(windows_size)
    kernel=make_kernel(window_shape, windows_size)

    #min
    img=minimum(image_array, kernel)

    #show image
    if im_show:
        show_image(img)




if __name__=="__main__":
    img=read_image(image_noise, im_show=True)
    #median_image(img, 3)
    #noise_filter_image(img, 3, "disk")
    #minimum_image(img, 3)
