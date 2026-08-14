import cv2
import numpy as np

img_path = "captures_8-13_three_flat/capture_52.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(gray)

K = np.array([
    [3601.38, 0.0, 2304.0],
    [0.0, 3601.38, 1296.0],
    [0.0, 0.0, 1.0]
])
dist = np.array([-0.5823, 7.9646, 0.0, 0.0, -29.288])

s = 15.0
obj_pts = np.array([
    [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]
], dtype=np.float64)

c14 = None
c20 = None

ids = ids.flatten()
for i in range(len(ids)):
    if ids[i] == 14:
        c14 = corners[i][0]
    elif ids[i] == 20:
        c20 = corners[i][0]

if c14 is not None and c20 is not None:
    _, rvec14, tvec14 = cv2.solvePnP(obj_pts, c14, K, dist)
    _, rvec20, tvec20 = cv2.solvePnP(obj_pts, c20, K, dist)
    
    dist_3d = np.linalg.norm(tvec14 - tvec20)
    print(f"Physical distance between M14 and M20 origins: {dist_3d:.2f} mm")
    
    R14, _ = cv2.Rodrigues(rvec14)
    P_m14 = R14.T @ (tvec20 - tvec14)
    print(f"M20 offset relative to M14 origin: X={P_m14[0][0]:.2f}, Y={P_m14[1][0]:.2f}, Z={P_m14[2][0]:.2f} mm")
