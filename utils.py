import cv2
import numpy as np

def check_opencv_version():
    """
    Print out the current OpenCV version.
    """
    print("OpenCV version:", cv2.__version__)

def auto_crop_black(image, threshold=5):
    """
    Crops away large uniform black regions from an image's edges.
    Returns the cropped image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return image[y0:y1+1, x0:x1+1]
