import cv2
import numpy as np

def alpha_blend(panorama, warped):
    """
    Overlap region is blended with a fixed alpha.
    Non-overlapping regions are directly copied.
    Returns the updated panorama.
    """
    pano_gray   = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped,   cv2.COLOR_BGR2GRAY)

    mask_pano   = pano_gray > 0
    mask_warped = warped_gray > 0

    # Overlap & exclusive
    overlap   = mask_pano & mask_warped
    only_warp = (~mask_pano) & mask_warped

    # Copy non-overlapping part
    panorama[only_warp] = warped[only_warp]

    alpha = 0.5
    panorama[overlap] = (
        alpha * panorama[overlap].astype(np.float32) +
        (1 - alpha) * warped[overlap].astype(np.float32)
    ).astype(np.uint8)

    return panorama



def laplacian_blend_entire(panorama, warped, levels=4):
    """
    Laplacian subregion blending approach:
      1) Find overlapping region between 'panorama' and 'warped'
      2) Build a half-split mask (left half=1, right half=0)
      3) Call the Laplacian pyramid routine on that subregion only
      4) Place the blended subregion back into 'panorama'
    """
    # Convert to grayscale for overlap detection
    pano_gray   = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped,   cv2.COLOR_BGR2GRAY)

    # Non-zero means valid region
    mask_pano   = (pano_gray   > 0)
    mask_warped = (warped_gray > 0)

    overlap   = mask_pano & mask_warped
    only_warp = (~mask_pano) & mask_warped

    # 1) Copy non-overlapping region directly
    panorama[only_warp] = warped[only_warp]

    # 2) If there's overlap, Laplacian-blend just that overlapping subregion
    overlap_coords = np.argwhere(overlap)
    if overlap_coords.size > 0:
        y_min, x_min = overlap_coords.min(axis=0)
        y_max, x_max = overlap_coords.max(axis=0)

        pano_sub   = panorama[y_min:y_max+1, x_min:x_max+1]
        warped_sub = warped[y_min:y_max+1, x_min:x_max+1]

        # Sanity check
        if pano_sub.shape != warped_sub.shape:
            raise ValueError(f"Subregion shapes differ! {pano_sub.shape} vs {warped_sub.shape}")

        h_sub, w_sub = pano_sub.shape[:2]

        # 3) Create a half-split mask: left half=1, right half=0
        #    (Adjust as desired for your specific scenario)
        mask_sub = np.zeros((h_sub, w_sub, 3), dtype=np.float32)
        mask_sub[:, : (w_sub // 2), :] = 1.0

        # 4) Run the actual Laplacian pyramid code
        blended_sub = _laplacian_pyramid_blending(pano_sub, warped_sub, mask_sub, levels=levels)

        # 5) Paste blended result back into the panorama
        panorama[y_min:y_max+1, x_min:x_max+1] = blended_sub

    return panorama


def _laplacian_pyramid_blending(img1, img2, mask, levels=6):
    """
    Actual multi-band Laplacian pyramid blending logic.
    - 'img1', 'img2', 'mask' must be the same shape.
    - This function is called internally by 'laplacian_blend_entire'.
    """
    # 1) Shape check
    if img1.shape != img2.shape or img1.shape != mask.shape:
        raise ValueError(f"Shapes differ: {img1.shape} vs {img2.shape} vs {mask.shape}")

    # 2) Convert a 2D mask to 3D if needed
    if mask.ndim == 2 and img1.ndim == 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask = mask.astype(np.float32)

    # 3) Build Gaussian pyramids (img1, img2, mask)
    gp1, gp2, gpM = [img1.astype(np.float32)], [img2.astype(np.float32)], [mask]
    for lvl in range(1, levels+1):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        gpM.append(cv2.pyrDown(gpM[-1]))

    # 4) Build Laplacian pyramids for img1 & img2
    lp1 = [gp1[-1]]  # smallest level
    lp2 = [gp2[-1]]
    for i in range(levels, 0, -1):
        size1 = (gp1[i-1].shape[1], gp1[i-1].shape[0])
        up1 = cv2.resize(cv2.pyrUp(gp1[i]), size1)
        L1 = cv2.subtract(gp1[i-1], up1)

        size2 = (gp2[i-1].shape[1], gp2[i-1].shape[0])
        up2 = cv2.resize(cv2.pyrUp(gp2[i]), size2)
        L2 = cv2.subtract(gp2[i-1], up2)

        lp1.append(L1)
        lp2.append(L2)

    # 5) Reverse the mask pyramid (or you could build a Laplacian for it as well)
    gpM.reverse()

    # 6) Blend each level
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gpM):
        blended_level = l1 * gm + l2 * (1 - gm)
        LS.append(blended_level)

    # 7) Reconstruct
    blended = LS[0]
    for i in range(1, len(LS)):
        up = cv2.pyrUp(blended)
        sizeLS = (LS[i].shape[1], LS[i].shape[0])
        up = cv2.resize(up, sizeLS)  # Ensure exact shape match
        blended = cv2.add(up, LS[i])

    return blended.astype(np.uint8)



def poisson_blend(panorama, warped):
    """
    Uses Poisson (seamless) cloning on the overlapping region.
    """
    pano_gray   = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped,   cv2.COLOR_BGR2GRAY)

    mask_pano   = (pano_gray   > 0)
    mask_warped = (warped_gray > 0)

    overlap   = mask_pano & mask_warped
    only_warp = (~mask_pano) & mask_warped

    # Place non-overlapping
    panorama[only_warp] = warped[only_warp]

    overlap_coords = np.argwhere(overlap)
    if overlap_coords.size > 0:
        y_min, x_min = overlap_coords.min(axis=0)
        y_max, x_max = overlap_coords.max(axis=0)

        pano_sub   = panorama[y_min:y_max+1, x_min:x_max+1]
        warped_sub = warped[y_min:y_max+1, x_min:x_max+1]

        if pano_sub.shape != warped_sub.shape:
            raise ValueError(f"Subregion shapes differ: {pano_sub.shape} vs {warped_sub.shape}")

        h_sub, w_sub = warped_sub.shape[:2]
        mask_sub = 255 * np.ones((h_sub, w_sub), dtype=np.uint8)
        center = (w_sub // 2, h_sub // 2)

        blended_sub = cv2.seamlessClone(
            src=warped_sub,
            dst=pano_sub,
            mask=mask_sub,
            p=center,
            flags=cv2.MIXED_CLONE
        )
        panorama[y_min:y_max+1, x_min:x_max+1] = blended_sub

    return panorama


def feather_blend(panorama, warped, axis='horizontal'):
    """
    Feather blend along the overlap. 
    'axis' = 'horizontal' or 'vertical'
    """
    pano_gray   = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped,   cv2.COLOR_BGR2GRAY)

    mask_pano   = (pano_gray   > 0)
    mask_warped = (warped_gray > 0)

    overlap   = mask_pano & mask_warped
    only_warp = (~mask_pano) & mask_warped

    # Place non-overlapping
    panorama[only_warp] = warped[only_warp]

    overlap_coords = np.argwhere(overlap)
    if overlap_coords.size > 0:
        y_min, x_min = overlap_coords.min(axis=0)
        y_max, x_max = overlap_coords.max(axis=0)

        pano_sub   = panorama[y_min:y_max+1, x_min:x_max+1]
        warped_sub = warped[y_min:y_max+1, x_min:x_max+1]

        if pano_sub.shape != warped_sub.shape:
            raise ValueError(f"Subregion shapes differ: {pano_sub.shape} vs {warped_sub.shape}")

        # Actually do the feathering
        blended_sub = _feather_subregion(pano_sub, warped_sub, axis)
        panorama[y_min:y_max+1, x_min:x_max+1] = blended_sub

    return panorama

def _feather_subregion(imgA, imgB, axis='horizontal'):
    """
    Both images must be same shape. We do a linear alpha ramp.
    """
    h, w = imgA.shape[:2]
    A = imgA.astype(np.float32)
    B = imgB.astype(np.float32)

    if axis == 'horizontal':
        alpha_col = np.linspace(1.0, 0.0, w, dtype=np.float32)
        alpha = np.tile(alpha_col, (h, 1))
    else:
        alpha_row = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
        alpha = np.tile(alpha_row, (1, w))

    alpha_3ch = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    out = A * alpha_3ch + B * (1.0 - alpha_3ch)
    return out.astype(np.uint8)
