import cv2
import numpy as np
from blending import alpha_blend, laplacian_blend_entire, poisson_blend, feather_blend

def stitch_with_custom_blending(images, H_to_ref, translation, width, height, blend_method="alpha"):
    """
    Warps each image to the final canvas based on H_to_ref and translation.
    Then calls the selected blending method to blend with the ongoing panorama.
    Returns the final panorama.
    """
    # Initialize blank panorama
    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        # Compose the translation with H_to_ref
        H_shifted = translation @ H_to_ref[i]
        # Warp image
        warped = cv2.warpPerspective(img, H_shifted, (width, height))

        # Choose blending method
        if blend_method == "alpha":
            panorama = alpha_blend(panorama, warped)
        elif blend_method == "laplacian":
            panorama = laplacian_blend_entire(panorama, warped, levels=6)
        elif blend_method == "poisson":
            panorama = poisson_blend(panorama, warped)
        elif blend_method == "feather":
            panorama = feather_blend(panorama, warped, axis='horizontal')
        else:
            raise ValueError(f"Unknown blend_method: {blend_method}")

    return panorama
