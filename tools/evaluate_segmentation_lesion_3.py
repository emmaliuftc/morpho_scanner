import os
import glob
import json
import time
import cv2
import numpy as np
import rembg
from PIL import Image
from scipy import ndimage

def main():
    img_dir = "nerf_160826_0824_8-16_lesion_3/images_4"
    existing_masks_dir = "nerf_160826_0824_8-16_lesion_3/masks_4"
    out_base = "evaluation_segmentation_lesion_3_0816"
    
    dirs = {
        "stage1": os.path.join(out_base, "stage1_u2net_rembg"),
        "stage2_green": os.path.join(out_base, "stage2_hsv_green_filter"),
        "stage3": os.path.join(out_base, "stage3_aruco_convex_hull"),
        "stage4": os.path.join(out_base, "stage4_morphological_cleanup"),
        "previews": os.path.join(out_base, "progression_previews"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    images = sorted(glob.glob(os.path.join(img_dir, "capture_*.png")), 
                    key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    total_imgs = len(images)
    print(f"Found {total_imgs} images in {img_dir}")
    print("Initializing rembg (u2net) session...")
    session = rembg.new_session("u2net")
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    metrics = []
    start_time = time.time()
    
    # Standard Green Parameters for Clay / Neutral Objects (Lesion 3 is purple/magenta/pink/white)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    for i, img_path in enumerate(images):
        t0 = time.time()
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 1. Stage 1: U2-Net Foreground Proposal
        s1_mask_path = os.path.join(dirs["stage1"], f"mask_{stem}.png")
        if os.path.exists(s1_mask_path):
            stage1_mask = cv2.imread(s1_mask_path, 0)
        else:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            rembg_out = rembg.remove(img_pil, session=session)
            alpha = np.array(rembg_out.split()[-1]) > 128
            stage1_mask = ndimage.binary_fill_holes(ndimage.binary_closing(alpha, iterations=5)).astype(np.uint8) * 255
            cv2.imwrite(s1_mask_path, stage1_mask)
        s1_pixels = int(np.count_nonzero(stage1_mask))
        s1_success = bool(s1_pixels > 5000)
        
        # 2. Stage 2: Standard HSV Green Exclusion (Turntable Platter)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_platter_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_platter_mask = cv2.bitwise_not(green_platter_mask)
        stage2_mask = ((stage1_mask > 0) & (non_platter_mask > 0)).astype(np.uint8) * 255
        
        s2_pixels = int(np.count_nonzero(stage2_mask))
        s2_pixels_removed = s1_pixels - s2_pixels
        s2_success = bool(s2_pixels > 5000)
        
        # 3. Stage 3: Geometric ArUco & Convex Hull Masking
        corners, ids, _ = detector.detectMarkers(img)
        num_markers = len(ids) if ids is not None else 0
        s3_aruco_detected = num_markers > 0
        
        aruco_mask = np.ones((h, w), dtype=np.uint8) * 255
        hull_area = 0.0
        if s3_aruco_detected:
            all_pts = []
            for corner_set in corners:
                for pt in corner_set[0]:
                    all_pts.append(pt)
            all_pts = np.array(all_pts, dtype=np.int32)
            hull = cv2.convexHull(all_pts)
            hull_area = float(cv2.contourArea(hull))
            cv2.fillConvexPoly(aruco_mask, hull, 0)
            kernel_11 = np.ones((11, 11), np.uint8)
            aruco_mask = cv2.erode(aruco_mask, kernel_11)
            
        stage3_mask = cv2.bitwise_and(stage2_mask, aruco_mask)
        s3_pixels = int(np.count_nonzero(stage3_mask))
        s3_board_pixels_removed = s2_pixels - s3_pixels
        s3_success = bool(s3_pixels > 5000)
        
        # 4. Stage 4: Morphological Opening, Infill, and Primary Component Isolation
        kernel_open = np.ones((3, 3), np.uint8)
        stage4_open = cv2.morphologyEx(stage3_mask, cv2.MORPH_OPEN, kernel_open)
        stage4_filled = ndimage.binary_fill_holes(stage4_open > 0).astype(np.uint8) * 255
        
        num_labels_before, _ = cv2.connectedComponents((stage3_mask > 0).astype(np.uint8))
        num_labels_intermediate, labels_im, stats, _ = cv2.connectedComponentsWithStats((stage4_filled > 0).astype(np.uint8))
        
        # Keep primary component (Lesion 3)
        if num_labels_intermediate > 1:
            largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            stage4_mask = (labels_im == largest_idx).astype(np.uint8) * 255
            num_labels_after = 2
        else:
            stage4_mask = stage4_filled
            num_labels_after = num_labels_intermediate
            
        s4_pixels = int(np.count_nonzero(stage4_mask))
        s4_noise_islands_removed = max(0, (num_labels_before - 1) - (num_labels_after - 1))
        s4_success = bool((num_labels_after - 1) == 1 and s4_pixels > 5000)
        
        # IoU comparison with existing mask if available
        existing_mask_path = os.path.join(existing_masks_dir, f"mask_{stem}.png")
        iou = None
        if os.path.exists(existing_mask_path):
            existing_mask = cv2.imread(existing_mask_path, 0)
            if existing_mask is not None:
                intersection = np.count_nonzero((stage4_mask > 0) & (existing_mask > 0))
                union = np.count_nonzero((stage4_mask > 0) | (existing_mask > 0))
                iou = round(float(intersection / max(1, union) * 100), 2)
                
        # Overall Pipeline Success
        pipeline_success = bool(s1_success and s2_success and s3_success and s4_success)
        
        # Save intermediate masks
        cv2.imwrite(os.path.join(dirs["stage1"], f"mask_{stem}.png"), stage1_mask)
        cv2.imwrite(os.path.join(dirs["stage2_green"], f"mask_{stem}.png"), stage2_mask)
        cv2.imwrite(os.path.join(dirs["stage3"], f"mask_{stem}.png"), stage3_mask)
        cv2.imwrite(os.path.join(dirs["stage4"], f"mask_{stem}.png"), stage4_mask)
        
        # 6-Panel Progression Preview (576 x 324 per panel with bottom caption)
        p_w = 576
        p_h = 324
        p_size = (p_w, p_h)
        
        p_orig = cv2.resize(img, p_size)
        p_s1 = cv2.cvtColor(cv2.resize(stage1_mask, p_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        p_s2 = cv2.cvtColor(cv2.resize(stage2_mask, p_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        p_s3 = cv2.cvtColor(cv2.resize(stage3_mask, p_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        p_s4 = cv2.cvtColor(cv2.resize(stage4_mask, p_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        
        # Object Cutout Overlay
        p_cutout = p_orig.copy()
        p_s4_bin = cv2.resize(stage4_mask, p_size, interpolation=cv2.INTER_NEAREST)
        p_cutout[p_s4_bin == 0] = p_cutout[p_s4_bin == 0] // 4
        
        panels = [p_orig, p_s1, p_s2, p_s3, p_s4, p_cutout]
        titles = [
            "1. Raw Calibrated",
            "2. S1: U2-Net Proposal",
            "3. S2: Standard Green",
            "4. S3: ArUco Hull",
            "5. S4: Clean Silhouette",
            "6. Lesion 3 Cutout"
        ]
        subtitles = [
            "Input Camera Frame",
            "Foreground AI Proposal",
            "Turntable Platter Excluded",
            "Board & Markers Excised",
            "Watertight Solid Mask",
            "Segmented Foreground"
        ]
        
        banner_h = 85
        combined_panels = []
        for panel, title, subtitle in zip(panels, titles, subtitles):
            banner = np.zeros((banner_h, p_w, 3), dtype=np.uint8)
            banner[:] = (18, 18, 20)
            cv2.line(banner, (0, 0), (p_w, 0), (55, 55, 60), 2)
            
            font_main = cv2.FONT_HERSHEY_DUPLEX
            scale_main = 1.0
            thick_main = 2
            (tw, th), _ = cv2.getTextSize(title, font_main, scale_main, thick_main)
            tx = (p_w - tw) // 2
            cv2.putText(banner, title, (tx, 40), font_main, scale_main, (255, 255, 255), thick_main, cv2.LINE_AA)
            
            font_sub = cv2.FONT_HERSHEY_SIMPLEX
            scale_sub = 0.58
            thick_sub = 1
            (sw, sh), _ = cv2.getTextSize(subtitle, font_sub, scale_sub, thick_sub)
            sx = (p_w - sw) // 2
            cv2.putText(banner, subtitle, (sx, 68), font_sub, scale_sub, (0, 230, 255), thick_sub, cv2.LINE_AA)
            
            combined_panels.append(np.vstack([panel, banner]))
            
        progression_row = np.hstack(combined_panels)
        cv2.imwrite(os.path.join(dirs["previews"], f"progression_{stem}.jpg"), progression_row)
        
        elapsed = round(time.time() - t0, 2)
        record = {
            "image": basename,
            "index": i,
            "stage1_pixels": s1_pixels,
            "stage1_success": s1_success,
            "stage2_pixels": s2_pixels,
            "stage2_platter_removed": s2_pixels_removed,
            "stage2_success": s2_success,
            "aruco_markers_detected": num_markers,
            "stage3_hull_area": hull_area,
            "stage3_pixels": s3_pixels,
            "stage3_board_removed": s3_board_pixels_removed,
            "stage3_success": s3_success,
            "stage4_final_pixels": s4_pixels,
            "stage4_islands_before": int(num_labels_before - 1),
            "stage4_islands_after": int(num_labels_after - 1),
            "stage4_islands_removed": int(s4_noise_islands_removed),
            "stage4_success": s4_success,
            "iou_with_existing": iou,
            "pipeline_success": pipeline_success,
            "elapsed_seconds": elapsed
        }
        metrics.append(record)
        iou_str = f" | IoU: {iou}%" if iou is not None else ""
        print(f"[{i+1:02d}/{total_imgs}] {basename} ({elapsed}s) | S1: {s1_pixels} | S2 Rem: {s2_pixels_removed} | ArUco: {num_markers} | S4: {s4_pixels}{iou_str} | Success: {pipeline_success}")

    total_elapsed = round(time.time() - start_time, 2)
    
    # Compute Aggregate Summary
    total_count = len(metrics)
    valid_ious = [m["iou_with_existing"] for m in metrics if m["iou_with_existing"] is not None]
    
    summary = {
        "dataset": img_dir,
        "total_frames": total_count,
        "total_time_seconds": total_elapsed,
        "avg_time_per_frame_seconds": round(total_elapsed / total_count, 2),
        "stage1_u2net": {
            "success_count": sum(1 for m in metrics if m["stage1_success"]),
            "success_rate_pct": round(sum(1 for m in metrics if m["stage1_success"]) / total_count * 100, 2),
            "avg_pixels": int(np.mean([m["stage1_pixels"] for m in metrics]))
        },
        "stage2_standard_green": {
            "success_count": sum(1 for m in metrics if m["stage2_success"]),
            "success_rate_pct": round(sum(1 for m in metrics if m["stage2_success"]) / total_count * 100, 2),
            "avg_pixels_removed": int(np.mean([m["stage2_platter_removed"] for m in metrics])),
            "avg_pixels_remaining": int(np.mean([m["stage2_pixels"] for m in metrics]))
        },
        "stage3_aruco_hull": {
            "success_count": sum(1 for m in metrics if m["stage3_success"]),
            "success_rate_pct": round(sum(1 for m in metrics if m["stage3_success"]) / total_count * 100, 2),
            "avg_markers_detected": round(float(np.mean([m["aruco_markers_detected"] for m in metrics])), 2),
            "avg_board_pixels_removed": int(np.mean([m["stage3_board_removed"] for m in metrics]))
        },
        "stage4_morphological_cleanup": {
            "success_count": sum(1 for m in metrics if m["stage4_success"]),
            "success_rate_pct": round(sum(1 for m in metrics if m["stage4_success"]) / total_count * 100, 2),
            "avg_islands_removed": round(float(np.mean([m["stage4_islands_removed"] for m in metrics])), 2),
            "avg_final_silhouette_pixels": int(np.mean([m["stage4_final_pixels"] for m in metrics]))
        },
        "comparison_with_baseline_masks": {
            "mean_iou_pct": round(float(np.mean(valid_ious)), 2) if len(valid_ious) > 0 else None,
            "min_iou_pct": round(float(np.min(valid_ious)), 2) if len(valid_ious) > 0 else None,
            "max_iou_pct": round(float(np.max(valid_ious)), 2) if len(valid_ious) > 0 else None
        },
        "end_to_end_pipeline": {
            "success_count": sum(1 for m in metrics if m["pipeline_success"]),
            "success_rate_pct": round(sum(1 for m in metrics if m["pipeline_success"]) / total_count * 100, 2)
        }
    }
    
    with open(os.path.join(out_base, "metrics_per_frame.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(out_base, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n==========================================")
    print(f"LESION 3 EVALUATION COMPLETE ({total_elapsed}s)")
    print(f"Stage 1 ($U^2$-Net) Success Rate: {summary['stage1_u2net']['success_rate_pct']}%")
    print(f"Stage 2 (Standard Green) Success Rate: {summary['stage2_standard_green']['success_rate_pct']}%")
    print(f"Stage 3 (ArUco Hull) Success Rate: {summary['stage3_aruco_hull']['success_rate_pct']}%")
    print(f"Stage 4 (Morph Cleanup) Success Rate: {summary['stage4_morphological_cleanup']['success_rate_pct']}%")
    print(f"End-to-End Pipeline Success Rate: {summary['end_to_end_pipeline']['success_rate_pct']}%")
    if summary['comparison_with_baseline_masks']['mean_iou_pct'] is not None:
        print(f"Mean IoU with Existing Masks: {summary['comparison_with_baseline_masks']['mean_iou_pct']}%")
    print(f"==========================================")

if __name__ == "__main__":
    main()
