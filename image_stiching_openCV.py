import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_crop_black(image, threshold=5):
    """
    Crops away large uniform black (near 0) regions from the edges.
    threshold: intensity below which we consider black.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image  # no valid region
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return image[y0:y1+1, x0:x1+1]


# 1) READ IMAGES
image_paths = ["1.jpg","2.jpg","3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg"]
images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Could not read {path}")
    images.append(img)

# 2) CREATE THE STITCHER
#    Note: If you have OpenCV 4+, use cv2.Stitcher_create().
#          For older versions, it might be cv2.createStitcher().
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# 3) STITCH
(status, stitched) = stitcher.stitch(images)

if status == cv2.STITCHER_OK:
    # 4) AUTO-CROP TO REMOVE BLACK BORDERS
    stitched_cropped = auto_crop_black(stitched, threshold=5)

    # 5) DISPLAY
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(stitched_cropped, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("OpenCV Stitcher Result ")
    plt.show()

#     # (Optional) SAVE
#     cv2.imwrite("final_panorama_openCV.jpg", stitched_cropped)
#     print("Panorama stitched and saved to final_panorama.jpg")
# else:
#     print(f"Stitching failed with error code = {status}")