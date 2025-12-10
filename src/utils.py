# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:29:25 2025

@author: bhejazi

Utility functions
"""
from __future__ import annotations
import json
import numpy as np

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z)
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """
    Return Euler Z-Y-X (yaw, pitch, roll) in radians from rotation matrix
    """
    if abs(R[2, 0]) < 1 - 1e-9:
        yaw = np.arctan2(R[1, 0], R[0, 0])      # Z
        pitch = np.arcsin(-R[2, 0])             # Y
        roll = np.arctan2(R[2, 1], R[2, 2])     # X
    else:
        # Gimbal lock fallback
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.pi / 2 * np.sign(-R[2, 0])
        roll = 0.0
    return np.array([yaw, pitch, roll])

def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous transform from R (3x3) and t (3,)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, :4][:, 3] = t.reshape(3)
    return T

def load_intrinsics(path: str) -> dict:
    """
    Load per-camera intrinsics JSON:
    {
      "cam0": {"K": [[...],[...],[...]], "distortion":[k1,k2,p1,p2,k3], "width":..., "height":...},
      ...
    }
    """
    with open(path, "r") as f:
        data = json.load(f)
    intr = {}
    for cam, v in data.items():
        intr[cam] = {
            "K": np.array(v["K"], float),
            "distortion": np.array(v.get("distortion", [0, 0, 0, 0, 0]), float).reshape(-1),
            "width": int(v["width"]),
            "height": int(v["height"]),
        }
    return intr

def load_observations(path: str) -> dict:
    """
    Load per-camera observations JSON:
    {
      "cameras": {
        "cam0": {"points_3d":[[x,y,z],...], "points_2d":[[u,v],...]},
        ...
      }
    }
    """
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for cam, v in data["cameras"].items():
        pts3d = np.array(v["points_3d"], float)
        pts2d = np.array(v["points_2d"], float)
        if pts3d.shape[0] != pts2d.shape[0]:
            raise ValueError(f"[{cam}] points_3d vs points_2d count mismatch")
        out[cam] = {"points_3d": pts3d, "points_2d": pts2d}
    return out

