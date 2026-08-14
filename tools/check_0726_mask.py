import cv2
import numpy as np
img = cv2.imread('captures_0726_nerf_dataset/capture_0.png')
mask = cv2.imread('captures_0726_nerf_dataset/mask_capture_0.png', cv2.IMREAD_GRAYSCALE)
if img is not None and mask is not None:
    # See if the mask covers the whole green plate, or just the object
    h, w = mask.shape
    img_rs = cv2.resize(img, (w, h))
    masked_img = cv2.bitwise_and(img_rs, img_rs, mask=mask)
    cv2.imwrite('/tmp/0726_masked_preview.png', masked_img)
    print("Mask max:", mask.max(), "min:", mask.min(), "mean:", mask.mean())
    print("Wrote /tmp/0726_masked_preview.png")
else:
    print("Failed to load images")
