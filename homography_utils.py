import numpy as np

def choose_reference(len_images):
    """
    Choose the middle image index as the reference.
    """
    return len_images // 2

def build_H_to_ref(num_images, ref_index):
    """
    Returns an array to store the homography for each image
    w.r.t. the reference image. The reference image is set to Identity.
    """
    H_to_ref = [None] * num_images
    H_to_ref[ref_index] = np.eye(3, dtype=np.float64)
    return H_to_ref

def get_transformed_corners(img, H):
    """
    Returns the transformed corners of an image given a homography matrix H.
    """
    import cv2
    import numpy as np
    h, w = img.shape[:2]

    corners = np.array([
        [0,   0,   1],
        [w-1, 0,   1],
        [0,   h-1, 1],
        [w-1, h-1, 1]
    ], dtype=np.float32)

    # Perform matrix multiplication
    transformed = (H @ corners.T).T
    # Normalize by last coordinate
    transformed /= transformed[:, 2:3]
    return transformed[:, :2]

def compute_bounding_box(images, H_to_ref):
    """
    Compute the bounding box that fits all warped images.
    Returns (width, height, translation_matrix).
    """
    import numpy as np

    all_corners = []
    for i, img in enumerate(images):
        H = H_to_ref[i]
        corners_t = get_transformed_corners(img, H)
        all_corners.append(corners_t)

    all_corners = np.vstack(all_corners)
    min_x, min_y = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    max_x, max_y = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    width  = max_x - min_x
    height = max_y - min_y

    # Create a translation matrix so all coordinates remain positive
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float64)

    return width, height, translation
