import cv2
import numpy as np
import json
import os
import glob

# Paths
img_dir = "data/three_flat_real_calib/images_4"
bad_out = "data/three_flat_real_calib/bad_z"
good_out = "data/three_flat_real_calib/good_z"
os.makedirs(bad_out, exist_ok=True)
os.makedirs(good_out, exist_ok=True)

# 1. Get GOOD calibration
with open("captures_8-13_three_flat_calibrated/calibration_results.json", "r") as f:
    good_cal = json.load(f)

# The good calibration
good_C = np.array(good_cal['plate_center_mm'], dtype=np.float64)
good_N = np.array(good_cal['plate_normal'], dtype=np.float64)

# To get the BAD calibration, we run the bad math here
# Bad offsets: 20.51, 20.74
# No 8-point enforcement
print("Computing bad trajectory...")
def get_bad_cal():
    # We need the original K and dist from good_cal, because the intrinsic calibration was the same
    K = np.array(good_cal['camera_matrix_K'])
    dist = np.array(good_cal['distortion_coefficients'])
    
    images = sorted(glob.glob("captures_8-13_three_flat/*.jpg"), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    
    # Board with bad offsets
    s = 15.0
    ox, oy = 20.51, 20.74
    m0 = np.array([[0,0,0],[s,0,0],[s,s,0],[0,s,0]], dtype=np.float32)
    m6 = np.array([[ox,oy,0],[ox+s,oy,0],[ox+s,oy+s,0],[ox,oy+s,0]], dtype=np.float32)
    board_pts = {14: m0, 20: m6}
    
    poses = {}
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            ids = ids.flatten()
            obj_f, img_f = [], []
            for mid in [14, 20]:
                if mid in ids:
                    c = corners[list(ids).index(mid)][0]
                    obj_f.append(board_pts[mid])
                    img_f.append(c.astype(np.float64))
            if obj_f:
                obj_all = np.vstack(obj_f)
                img_all = np.vstack(img_f)
                ok, rvec, tvec = cv2.solvePnP(obj_all, img_all, K, dist)
                if ok and tvec.ravel()[2] > 0: # BAD: accepts 4 pts!
                    poses[idx] = tvec.ravel()
                    
    # SVD initial guess
    tvecs_raw = {i: t for i,t in poses.items() if 100 < t[2] < 800}
    tvecs_arr = np.array(list(tvecs_raw.values()))
    mean_t = tvecs_arr.mean(axis=0)
    centered = tvecs_arr - mean_t
    _, _, Vt = np.linalg.svd(centered)
    n = Vt[2]
    if n[2] < 0: n = -n
    # Center projection
    u_ax, v_ax = Vt[0], Vt[1]
    pu = centered @ u_ax
    pv = centered @ v_ax
    A = np.column_stack([2*pu, 2*pv, np.ones(len(pu))])
    B = pu**2 + pv**2
    uc, vc, _ = np.linalg.lstsq(A, B, rcond=None)[0]
    C = mean_t + uc * u_ax + vc * v_ax
    
    return C, n

bad_C, bad_N = get_bad_cal()
print(f"Bad C: {bad_C}")
print(f"Bad N: {bad_N}")

def draw_vis(img_in, out_path, C_rot, normal, K_scaled):
    img = cv2.imread(img_in)
    rvec0, tvec0 = np.zeros((3,1)), np.zeros((3,1))
    dist_zero = np.zeros(5)
    
    center_2d, _ = cv2.projectPoints(C_rot.reshape(1,1,3).astype(np.float64), rvec0, tvec0, K_scaled, dist_zero)
    cx, cy = int(round(center_2d[0][0][0])), int(round(center_2d[0][0][1]))
    
    s_vals = np.linspace(-40, 120, 300)
    z_pts_3d = np.array([C_rot + s * normal for s in s_vals], dtype=np.float64)
    z_pts_2d, _ = cv2.projectPoints(z_pts_3d, rvec0, tvec0, K_scaled, dist_zero)
    z_pts_px = z_pts_2d.reshape(-1, 2).astype(np.int32)
    
    arrow_3d = (C_rot + 80 * normal).reshape(1,1,3).astype(np.float64)
    arrow_2d, _ = cv2.projectPoints(arrow_3d, rvec0, tvec0, K_scaled, dist_zero)
    ax, ay = int(round(arrow_2d[0][0][0])), int(round(arrow_2d[0][0][1]))
    
    for j in range(0, len(z_pts_px) - 2, 4):
        cv2.line(img, tuple(z_pts_px[j]), tuple(z_pts_px[min(j+2, len(z_pts_px)-1)]), (255,255,0), 2, cv2.LINE_AA)
        
    cv2.arrowedLine(img, (cx, cy), (ax, ay), (255,255,0), 3, cv2.LINE_AA, tipLength=0.12)
    cv2.circle(img, (cx, cy), 15, (0,255,0), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, (0,0,255), -1, cv2.LINE_AA)
    
    cv2.imwrite(out_path, img)

K_scaled = np.array(good_cal['camera_matrix_K'])
K_scaled[:2, :] /= 4.0

images_4 = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

print("Rendering images...")
for i, p in enumerate(images_4):
    bname = os.path.basename(p)
    draw_vis(p, os.path.join(bad_out, bname), bad_C, bad_N, K_scaled)
    draw_vis(p, os.path.join(good_out, bname), good_C, good_N, K_scaled)

print("Done generating good/bad vis!")
