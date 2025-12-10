# Multi-camera extrinsics calibration

Workflow to compute camera extrinsics, position and orientation in a common world coordinate system, for a multi-camera setup using:

- Known intrinsics (camera matrix `K`, distortion coefficients, resolution)
- 2D–3D correspondences from calibration images


## Features
- Computes extrinsics using OpenCV's PnP algorithm
- Outputs:
  - Rotation & translation matrices
  - Camera position in world coordinates
  - Quaternion and Euler angles
  - RMS reprojection error


## Setup Instructions

### 1. Clone or download the repository

### 2. Create and activate a virtual environment
```
conda create -n camcal python=3.10 -y
conda activate camcal
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---


### 4. Run the calibration script
```bash
python src/calibrate_extrinsics.py --intrinsics data/intrinsics.json --observations data/observation.json --out_json output/extrinsics.json --out_csv output/extrinsics.csv --pnp ITERATIVE
```

Options:
- `pnp ITERATIVE` (default) or `EPnP`

## Inputs

### `intrinsics.json` format:
```json
{
  "cam0": {
    "K": [[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]],
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    "width": 1920,
    "height": 1080
  },
  "cam1": { ... },
  "cam2": { ... },
  ...
}
```

### `observations.json` format:
```json
{
  "cameras": {
    "cam0": {
      "points_3d": [[-0.16, -0.10, 3.0], ...],
      "points_2d": [[936.64, 594.48], ...]
    },
    "cam1": { ... },
    "cam2": { ... },
    ...
  }
}
```

## Outputs
- `output/extrinsics.json`: Detailed extrinsics per camera
- `output/extrinsics.csv`: Summary table with RMS error and camera positions

Example JSON snippet:
```json
{
  "extrinsics": {
    "cam0": {
      "R_cam_from_world": [[...],[...],[...]],
      "t_cam_from_world": [...],
      "R_world_from_cam": [[...],[...],[...]],
      "t_world_from_cam": [...],
      "quaternion_world_from_cam_wxyz": [...],
      "euler_zyx_world_from_cam_radians": [...],
      "metrics": {"rms_reproj_error_px": 0.85, "num_points": 54}
    }
  }
}
```



## Notes
- RMS error < 2 px indicates good calibration.
- Distortion coefficients follow OpenCV's Brown model: `[k1, k2, p1, p2, k3]`.

## Resources
- [OpenCV solvePnP](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Rodrigues formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

