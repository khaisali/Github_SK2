
# Functions used to manipulate images in OpenCV and PIL's Image.

import cv2
import numpy as np
from PIL import Image


def image_as_nparray(image):
    """
    Converts PIL's Image to numpy's array.
    - image: PIL's Image object.
    - Numpy's array of the image.
    """
    return np.asarray(image)


def nparray_as_image(nparray, mode='RGB'):
    """
    Converts numpy's array of image to PIL's Image.
    - nparray: Numpy's array of image.
    - mode: Mode of the conversion. Defaults to 'RGB'.
     PIL's Image containing the image is returned.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)


def load_image(source_path):
    """
    Loads RGB image and converts it to grayscale.
    - source_path: Image's source path.
    There returns the Image loaded from the path and converted to grayscale.
    """
    source_image = cv2.imread(source_path)
    final_pic = cv2.resize(source_image, (1000, 1000), fx=5.5, fy=5.5, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(final_pic, cv2.COLOR_BGR2GRAY)


def draw_with_alpha(source_image, image_to_draw, coordinates):
    """
    Draws a partially transparent image over another image.
    - source_image: Image to draw over.
    - image_to_draw: Image to draw.
    - coordinates: Coordinates to draw an image at. Tuple of x, y, width and height.
    """
    x, y, w, h = coordinates
    image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)
