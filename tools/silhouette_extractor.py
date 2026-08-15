import cv2
import numpy as np
import rembg
from PIL import Image
from scipy import ndimage

class SilhouetteExtractor:
    def __init__(self, model_name="u2net"):
        """Initializes the background removal model session once."""
        print(f"Initializing background removal model ({model_name})...")
        self.session = rembg.new_session(model_name)
        
    def get_silhouette_mask(self, img, center_2d=None, plate_radius_pixels=None):
        """
        Isolates the clay object by masking out the background, green plate, and ChArUco board.
        Returns a binary mask where 255 is the object, 0 is background.
        """
        height, width = img.shape[:2]
        
        # 1. rembg Mask: Isolate foreground elements (clay sculpture and ChArUco board)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rembg_out = rembg.remove(img_pil, session=self.session)
        alpha = np.array(rembg_out.split()[-1]) > 128
        filled = ndimage.binary_fill_holes(ndimage.binary_closing(alpha, iterations=5))
        
        # 2. HSV Color Mask: Remove green plate paper background highlights
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)
        
        # Combine rembg with green color exclusion
        obj_mask = (filled & (non_green_mask > 0)).astype(np.uint8) * 255
        
        # 4. ChArUco Board Mask: Detect markers and mask out the entire board using convex hull + dilation
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(img)
        
        aruco_mask = np.ones((height, width), dtype=np.uint8) * 255
        if ids is not None and len(ids) > 0:
            all_pts = []
            for corner_set in corners:
                for pt in corner_set[0]:
                    all_pts.append(pt)
            all_pts = np.array(all_pts, dtype=np.int32)
            
            hull = cv2.convexHull(all_pts)
            cv2.fillConvexPoly(aruco_mask, hull, 0)
            
            # Dilate the board mask slightly to completely wipe out outer white squares/margins
            kernel = np.ones((41, 41), np.uint8)
            aruco_mask = cv2.erode(aruco_mask, kernel)
            
        # 5. Final Combined Mask
        final_mask = cv2.bitwise_and(obj_mask, aruco_mask)
        
        # Morphological opening to clean up small floating pixels
        kernel_open = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        final_mask = ndimage.binary_fill_holes(final_mask > 0).astype(np.uint8) * 255
        
        return final_mask
