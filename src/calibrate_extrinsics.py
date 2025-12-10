# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:29:25 2025

@author: bhejazi

Calibrate camera extrinsics from 2D-3D correspondences and known intrinsics.

Outputs per camera:
- Camera-from-world and world-from-camera extrinsics (4×4)
- Position (camera center) and orientation (matrix, quaternion, Euler ZYX)
- Reprojection error statistics
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from utils import (load_intrinsics, load_observations, matrix_to_quaternion, matrix_to_euler_zyx, make_homogeneous)

def compute_extrinsics_for_camera(K: np.ndarray, dist: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray, method: str = "ITERATIVE"):
    """
    Solve for camera pose using OpenCV's PnP and compute reprojection error

    Returns:
        R (3x3), t (3,), rms_error (float), num_points (int)
    """
    # OpenCV expects float32
    objp = pts3d.astype(np.float32)
    imgp = pts2d.astype(np.float32)

    # Choose PnP method
    flag = cv2.SOLVEPNP_ITERATIVE if method.upper() == "ITERATIVE" else cv2.SOLVEPNP_EPNP

    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
    if not success:
        raise RuntimeError("solvePnP failed")

    # Optional local refinement (OpenCV >= 4.1)
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, K, dist, rvec, tvec)
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Compute reprojection errors
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    errs = np.linalg.norm(imgp - proj, axis=1)
    rms = float(np.sqrt(np.mean(errs ** 2)))
    return R, tvec.reshape(3), rms, int(len(errs))

def main():
    ap = argparse.ArgumentParser(description="Compute camera extrinsics from 2D–3D correspondences and intrinsics.")
    ap.add_argument("--intrinsics", required=True, help="Path to intrinsics.json")
    ap.add_argument("--observations", required=True, help="Path to observation.json")
    ap.add_argument("--out_json", default="output/extrinsics.json", help="Output JSON path")
    ap.add_argument("--out_csv", default="output/extrinsics.csv", help="Output CSV path")
    ap.add_argument("--pnp", default="ITERATIVE", choices=["ITERATIVE", "EPnP"], help="PnP solver type")
    args = ap.parse_args()

    intr = load_intrinsics(args.intrinsics)
    obs = load_observations(args.observations)

    ex_all = {}
    rows = []

    for cam, ci in intr.items():
        if cam not in obs:
            print(f"[WARN] No observations found for camera '{cam}'. Skipping.")
            continue

        pts3d = obs[cam]["points_3d"]
        pts2d = obs[cam]["points_2d"]
        if pts3d.shape[0] < 4:
            print(f"[WARN] Camera '{cam}' has only {pts3d.shape[0]} points; pose may be unstable. Skipping.")
            continue

        R, t, rms, n = compute_extrinsics_for_camera(ci["K"], ci["distortion"], pts3d, pts2d, args.pnp)

        # Camera position in world coordinates
        C = -R.T @ t
        # Package outputs (both conventions)
        ex_all[cam] = {
            "R_cam_from_world": R.tolist(),
            "t_cam_from_world": t.tolist(),
            "R_world_from_cam": R.T.tolist(),
            "t_world_from_cam": C.tolist(),
            "T_cam_from_world_4x4": make_homogeneous(R, t).tolist(),
            "T_world_from_cam_4x4": make_homogeneous(R.T, C).tolist(),
            "quaternion_world_from_cam_wxyz": matrix_to_quaternion(R.T).tolist(),
            "euler_zyx_world_from_cam_radians": matrix_to_euler_zyx(R.T).tolist(),
            "metrics": {"num_points": n, "rms_reproj_error_px": rms}
        }

        rows.append(
            {
                "camera": cam,
                "num_points": n,
                "rms_error_px": rms,
                "position_world_x": float(C[0]),
                "position_world_y": float(C[1]),
                "position_world_z": float(C[2]),
            }
        )

    # Ensure parent directory exists
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Write outputs
    with open(args.out_json, "w") as f:
        json.dump({"extrinsics": ex_all}, f, indent=2)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[OK] Saved: {args.out_json}")
    print(f"[OK] Saved: {args.out_csv}")

if __name__ == "__main__":
    main()
