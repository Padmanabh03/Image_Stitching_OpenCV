import cv2

def create_flann_matcher():
    """
    Create and return a FLANN-based matcher with recommended parameters.
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)

def find_homography(kp1, des1, kp2, des2, flann, lowe_ratio=0.7):
    """
    Finds homography using FLANN-based matching + Lowe's ratio test.
    Returns (H, good_matches).
    If not enough good matches, returns (None, None).
    """
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]

    print(f"Found {len(good_matches)} good matches.")

    if len(good_matches) > 10:
        import numpy as np
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, good_matches
    else:
        return None, None
