import cv2
import numpy as np

from image_loader import load_images
from feature_detection import detect_features
from feature_matching import create_flann_matcher, find_homography
from homography_utils import choose_reference, build_H_to_ref, compute_bounding_box
from utils import check_opencv_version, auto_crop_black
from stitching import stitch_with_custom_blending
from plot_utils import show_image

def main():
    # 1) Check OpenCV version
    check_opencv_version()

    # 2) Set up image paths
    image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg"]

    # 3) Load images with a certain downscale factor (0.5 or 1.0, etc.)
    scale = 0.5
    images = load_images(image_paths, scale=scale)

    # 4) Detect keypoints/descriptors
    kd_list = detect_features(images)  # list of (kp, des) pairs

    # 5) Build homographies relative to reference
    ref_index = choose_reference(len(images))  # typically middle image
    H_to_ref = build_H_to_ref(len(images), ref_index)

    # 6) Create FLANN matcher
    flann = create_flann_matcher()

    # 7) For images to the left of reference
    for i in range(ref_index - 1, -1, -1):
        H_i_ip1, _ = find_homography(
            kd_list[i][0], kd_list[i][1],
            kd_list[i+1][0], kd_list[i+1][1],
            flann
        )
        if H_i_ip1 is not None:
            H_to_ref[i] = H_to_ref[i+1] @ H_i_ip1
        else:
            print(f"Warning: Could not find homography between image {i} and {i+1}")
            H_to_ref[i] = np.eye(3, dtype=np.float64)

    # 8) For images to the right of reference
    for i in range(ref_index + 1, len(images)):
        H_i_im1, _ = find_homography(
            kd_list[i][0], kd_list[i][1],
            kd_list[i-1][0], kd_list[i-1][1],
            flann
        )
        if H_i_im1 is not None:
            H_to_ref[i] = H_to_ref[i-1] @ H_i_im1
        else:
            print(f"Warning: Could not find homography between image {i} and {i-1}")
            H_to_ref[i] = np.eye(3, dtype=np.float64)

    # 9) Compute bounding box, get translation
    width, height, translation = compute_bounding_box(images, H_to_ref)
    print(f"Canvas size => width={width}, height={height}")

    # 10) Choose a blending method
    blend_method = input("Choose blending method (alpha/laplacian/poisson/feather): ").strip()

    # 11) Perform stitching
    panorama = stitch_with_custom_blending(images, H_to_ref, translation, width, height, blend_method)

    # 12) Auto-crop black
    final_panorama = auto_crop_black(panorama, threshold=5)

    # 13) Display result
    show_image(final_panorama, f"Final Panorama ({blend_method})")

    # 14) Save
    # out_name = f"final_panorama_{blend_method}.jpg"
    # cv2.imwrite(out_name, final_panorama)
    # print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
