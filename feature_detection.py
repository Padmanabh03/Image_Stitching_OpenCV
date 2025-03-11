import cv2

def detect_features(images):
    """
    Detect keypoints and descriptors for each image using SIFT.
    Returns a list of (keypoints, descriptors) for each image.
    """
    sift = cv2.SIFT_create()
    keypoints_descriptors = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints_descriptors.append((kp, des))
    return keypoints_descriptors
