import cv2
import numpy as np
import glob
import scipy.optimize as optimize

images = glob.glob("captures_8-13_three_flat/*.jpg")
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

K = np.array([[3601.38, 0, 2304.0], [0, 3601.38, 1296.0], [0, 0, 1]])
dist = np.array([-0.5823, 7.9646, 0.0, 0.0, -29.288])
s = 15.0

detections = []
for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        ids = ids.flatten()
        if 14 in ids and 20 in ids:
            c14 = corners[list(ids).index(14)][0]
            c20 = corners[list(ids).index(20)][0]
            detections.append((c14, c20))

print(f"Found {len(detections)} frames with both markers.")

def get_board(ox, oy):
    m14 = np.array([[0,0,0], [s,0,0], [s,s,0], [0,s,0]], dtype=np.float32)
    m20 = np.array([[ox,oy,0], [ox+s,oy,0], [ox+s,oy+s,0], [ox,oy+s,0]], dtype=np.float32)
    return np.vstack([m14, m20])

def cost_fn(params):
    ox, oy = params
    obj = get_board(ox, oy)
    errs = []
    for c14, c20 in detections:
        imgp = np.vstack([c14, c20]).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist)
        if not ok:
            return np.inf
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errs.extend((imgp.reshape(-1,2) - proj.reshape(-1,2)).flatten())
    return np.array(errs)

res = optimize.least_squares(cost_fn, [42.0, 20.0])
print(f"Optimized Offset: X={res.x[0]:.4f}, Y={res.x[1]:.4f}")
