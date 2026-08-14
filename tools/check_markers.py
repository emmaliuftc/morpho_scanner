import cv2
import glob

img_path = "captures_8-13_calibration/capture_0.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None:
    print(f"Detected {len(ids)} markers: {ids.flatten()}")
else:
    print("No markers detected.")
