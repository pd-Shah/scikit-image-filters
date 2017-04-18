from skimage import data, io, filters
from skimage.morphology import square, disk, rectangle
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
        io.imshow(image)
        io.show()
    return image

def low_pass_filter(image_array, window_size, center=8, other=-1, window_shape="square" ,im_show=False):
    """A low pass filter is the basis for most smoothing methods. An image is smoothed by decreasing the disparity between pixel values by averaging nearby pixels

    Parameters
    ----------
    image_array:numpy.array:
        input image

    im_show : Bool
        if True show result

    Other Parameters
    ----------------
    window_size:int
        the size of window

    center:int
        the value of c in center:
            1 1 1
            1 c 1
            1 1 1

    other:int
        the value of ones in below:
        disk shape:
            [0 0 1 0 0]
            [0 1 1 1 0]
            [1 1 c 1 1]
            [0 1 1 1 0]
            [0 0 1 0 0]
        square shape:
            [1 1 1]
            [1 c 1]
            [1 1 1]

    window_shape: str
        str is element from dict:{square, disk}


    Returns
    -------
    image : numpy.array
        noise filter array
    """

    #make kernel windows
    kernel=_make_kernel(window_size, center, other, window_shape)

    #image size x,y
    image_Length=len(image_array[1,:])
    image_width= len(image_array[:,1])

    #make kernel as shape as image_array
    x=ceil(image_Length/window_size)
    y=ceil(image_width/window_size)
    #repeat kernel as size of image
    kernel=np.tile(kernel,(x,y))
    #cut kernel to fit to image
    kernel=kernel[:image_Length,:image_width]

    #multiple kernel and windows
    image=image*kernel

    if im_show:
        io.imshow(image)
        io.show()
    return image

def _make_kernel(window_size, center=8, other=-1, window_shape="square"):
    """make kernel/window

    Parameters
    ----------
    window_size:int
        the size of window

    center:int
        the value of c in center:
            1 1 1
            1 c 1
            1 1 1

    other:int
        the value of ones in below:
        disk shape:
            [0 0 1 0 0]
            [0 1 1 1 0]
            [1 1 c 1 1]
            [0 1 1 1 0]
            [0 0 1 0 0]
        square shape:
            [1 1 1]
            [1 c 1]
            [1 1 1]

    window_shape: str
        str is element from dict:{square, disk}

    Other Parameters
    ----------------


    Returns
    -------
    window : numpy.array
        kernel or window
    """
    #check window_size be odd
    if window_size%2==0:
        raise KernelShapeError("window_size must be odd.")

    #check window shape types and make it
    if not window_shape in ["square", "disk"]:
        raise KernelShapeError("window_shape is unknown. window_shape={square, disk}.")
    elif window_shape=="square":
        window=square(window_size)
    elif window_shape=="disk":
        window=disk(window_size)

    #window size to float
    window=window.astype(float)

    #set center of array
    window_center_index=int(len(window)/2)
    window[window_center_index,window_center_index]=center

    #set others:
    window[window==1]=other

    return window


if __name__=="__main__":
    img=read_image(image_noise, im_show=True)
    #low_pass_filter(img, im_show=True)
    p=_make_kernel(3, window_shape="disk")
    print(p)
