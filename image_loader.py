import cv2

def load_images(image_paths, scale=0.5):
    """
    Loads images from paths and optionally resizes them by `scale`.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Could not read image {path}")
        # Downsample if desired
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(img)

    return images

