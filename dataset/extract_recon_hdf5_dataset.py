#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# =============================
# Image decode
# =============================

def bytes2im(x: Any) -> np.ndarray:
    """
    Compatible with NumPy 1.x and 2.x:
    - hdf5 may store images as scalar bytes (np.bytes_ / bytes) containing encoded png/jpg
    - or store as uint8 arrays directly
    """
    # h5py often returns numpy scalar/array; unwrap scalar 0-d array
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x[()]  # -> python scalar (bytes / str / number)

    # NumPy 2.0 removed np.string_; use np.bytes_ for bytes scalars
    if isinstance(x, np.bytes_):
        x = bytes(x)

    # In some datasets it might already be python bytes/bytearray
    if isinstance(x, (bytes, bytearray)):
        img = Image.open(BytesIO(x))
        return np.array(img)

    # If it's already an image array (HxW or HxWxC)
    if isinstance(x, np.ndarray):
        return x

    # Rare: h5py may give memoryview
    if isinstance(x, memoryview):
        img = Image.open(BytesIO(x.tobytes()))
        return np.array(img)

    # Some h5py versions may return objects with tobytes()
    if hasattr(x, "tobytes"):
        img = Image.open(BytesIO(x.tobytes()))
        return np.array(img)

    raise TypeError(f"Unsupported image type: {type(x)}")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_finite_latlon(latlon: np.ndarray) -> bool:
    latlon = np.asarray(latlon).reshape(-1)
    return latlon.size >= 2 and np.isfinite(latlon[0]) and np.isfinite(latlon[1])


def list_hdf5_files(input_dir: Path) -> List[Path]:
    """
    Robust file listing:
    - supports .h5/.hdf5 and case variants
    """
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".h5", ".hdf5"):
            files.append(p)
    return sorted(files)


# =============================
# Satellite geo reference
# =============================

class SatelliteGeoRef:
    """
    Local satellite image + (NW, SE) lat/lon -> linear lat/lon -> pixel mapping.
    No Google API, no zoom needed.
    """
    def __init__(self, image_path, nw_latlong, se_latlong):
        img = Image.open(image_path)
        self.W, self.H = img.size
        self.lat_n, self.lon_w = float(nw_latlong[0]), float(nw_latlong[1])
        self.lat_s, self.lon_e = float(se_latlong[0]), float(se_latlong[1])

        if abs(self.lon_e - self.lon_w) < 1e-12 or abs(self.lat_n - self.lat_s) < 1e-12:
            raise ValueError("Invalid NW/SE latlon for satellite mapping.")

    def latlon_to_pixel(self, lat, lon):
        x = (lon - self.lon_w) / (self.lon_e - self.lon_w) * (self.W - 1)
        y = (self.lat_n - lat) / (self.lat_n - self.lat_s) * (self.H - 1)
        return float(x), float(y)

    def in_map(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H


# =============================
# GPS → ENU
# =============================

def latlon_to_enu(lat, lon, lat0, lon0):
    """
    ENU meters (east, north). Small area approximation.
    """
    R = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    east = R * dlon * math.cos(math.radians(lat0))
    north = R * dlat
    return east, north


# =============================
# Camera model (K + T_wc)
# =============================

def build_T_wc_from_enu_yaw(east: float, north: float, up: float, yaw_rad: float) -> np.ndarray:
    """
    Build camera-to-world transform T_wc (4x4).

    World frame: ENU (x=east, y=north, z=up), meters
    Camera frame: OpenCV (x right, y down, z forward)
    yaw_rad: compass bearing radians, 0=north, +clockwise

    Assumption: no roll/pitch; camera forward aligns with heading.
    """
    forward_w = np.array([math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)   # z_cam in world
    right_w   = np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0], dtype=np.float64)  # x_cam in world
    up_w      = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_cam_w = right_w
    y_cam_w = -up_w
    z_cam_w = forward_w

    R_wc = np.stack([x_cam_w, y_cam_w, z_cam_w], axis=1)  # columns = camera axes in world
    C_w = np.array([east, north, up], dtype=np.float64)

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = C_w
    return T_wc


def guess_intrinsics(H, W):
    f = float(max(H, W))
    return np.array([[f, 0, W / 2],
                     [0, f, H / 2],
                     [0, 0, 1]], dtype=np.float64)


# =============================
# Main extraction
# =============================

def extract_one_hdf5(args, h5_path: Path, sat: Optional[SatelliteGeoRef]) -> Dict[str, Any]:
    """
    For ONE hdf5:
      - create a folder under output_dir named as h5_path.stem
      - save images into that folder
      - return a json payload (dict) for that hdf5
    """
    img_key = args.img_key
    gps_key = args.gps_key
    yaw_key = args.yaw_key

    # per-hdf5 output folder
    scene_dir = Path(args.output_dir) / h5_path.stem
    safe_mkdir(scene_dir)

    # optional: write a small info file to track provenance
    # (not required; comment out if you dislike extra files)
    # (we still keep "hdf5" field in json anyway)
    # (this is safe even if hdf5 is big; it's just a text file)
    with open(scene_dir / "source_hdf5.txt", "w", encoding="utf-8") as f:
        f.write(str(h5_path))

    records = []
    origin = None  # (lat0, lon0) per hdf5

    with h5py.File(h5_path, "r") as h5:
        # length: prefer collision/any else use image length
        if "collision/any" in h5:
            length = len(h5["collision/any"])
        elif img_key in h5:
            length = len(h5[img_key])
        else:
            return {
                "hdf5": str(h5_path),
                "scene": h5_path.stem,
                "error": f"skip: no collision/any and no {img_key}",
                "count": 0,
                "origin_latlon": None,
                "records": [],
            }

        frame_iter = range(0, length, args.every_n)

        for i in tqdm(
            frame_iter,
            desc=f"Frames ({h5_path.name})",
            unit="frame",
            leave=False,
        ):
            if args.max_frames_per_hdf5 > 0 and (i // args.every_n) >= args.max_frames_per_hdf5:
                break

            if gps_key not in h5 or yaw_key not in h5 or img_key not in h5:
                continue

            latlon = h5[gps_key][i]
            if not is_finite_latlon(latlon):
                continue

            lat, lon = float(latlon[0]), float(latlon[1])
            yaw = float(h5[yaw_key][i])

            # If yaw is degrees (0~360), uncomment:
            # yaw = math.radians(yaw)

            if origin is None:
                origin = (lat, lon)

            east, north = latlon_to_enu(lat, lon, origin[0], origin[1])
            up = 0.0

            img = bytes2im(h5[img_key][i])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            H, W = img.shape[:2]

            # Save image into per-hdf5 folder
            img_fname = f"{i:06d}.png"
            img_path = scene_dir / img_fname
            Image.fromarray(img).save(img_path)

            K = guess_intrinsics(H, W)
            T_wc = build_T_wc_from_enu_yaw(east, north, up, yaw)

            rec = {
                "timestep": int(i),
                "image": img_fname,  # relative to scene_dir
                "gps": {"lat": lat, "lon": lon, "alt": 0.0},
                "yaw_rad": float(yaw),
                "world_enu_m": {"east": float(east), "north": float(north), "up": float(up)},
                "camera": {
                    "K": K.tolist(),          # 3x3
                    "T_wc": T_wc.tolist(),    # 4x4 (camera -> world)
                    "camera_frame": "opencv_x_right_y_down_z_forward",
                    "world_frame": "ENU_m",
                    "extrinsic_equation": "X_world = T_wc * X_cam",
                    "yaw_convention": "compass_bearing_rad_0=north_cw+",
                },
            }

            if sat is not None:
                x, y = sat.latlon_to_pixel(lat, lon)
                rec["map_pixel"] = {"x": x, "y": y}
                rec["in_map"] = bool(sat.in_map(x, y))

            records.append(rec)

    payload = {
        "version": 1,
        "hdf5": str(h5_path),
        "scene": h5_path.stem,
        "count": len(records),
        "origin_latlon": None if origin is None else {"lat": origin[0], "lon": origin[1]},
        "records": records,
    }
    return payload


def extract_dataset(args):
    out_root = Path(args.output_dir)
    safe_mkdir(out_root)

    # Optional satellite mapping
    sat = None
    if args.sat_img:
        if not os.path.exists(args.sat_img):
            raise FileNotFoundError(f"--sat_img not found: {args.sat_img}")
        sat = SatelliteGeoRef(
            args.sat_img,
            (args.sat_nw_lat, args.sat_nw_lon),
            (args.sat_se_lat, args.sat_se_lon),
        )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    h5_files = list_hdf5_files(input_dir)
    print(f"Found {len(h5_files)} hdf5 files in {input_dir.resolve()}")

    # For convenience, also write an index json at output root
    index = {
        "version": 1,
        "input_dir": str(input_dir),
        "output_dir": str(out_root),
        "scenes": [],  # list of {"scene": ..., "dir": ..., "json": ...}
    }

    for h5_path in tqdm(h5_files, desc="HDF5 files", unit="file"):
        scene_name = h5_path.stem
        scene_dir = out_root / scene_name
        safe_mkdir(scene_dir)

        payload = extract_one_hdf5(args, h5_path, sat)

        # write per-hdf5 json next to images
        scene_json_path = scene_dir / args.json_name
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        index["scenes"].append({
            "scene": scene_name,
            "dir": scene_name,
            "json": str((scene_json_path).relative_to(out_root)),
            "count": payload.get("count", 0),
        })

    # write index
    with open(out_root / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total = sum(s["count"] for s in index["scenes"])
    print(f"Done. Total frames: {total}")
    print(f"Per-hdf5 folders are under: {out_root}")
    print(f"Wrote dataset index: {out_root / 'index.json'}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_dir", default="/data/tlxd/Cross-View/SCWM/tmp_test/")
    ap.add_argument("--output_dir", default="/data/tlxd/Cross-View/SCWM/tri_plane_debug/")
    ap.add_argument("--json_name", default="metadata.json")

    ap.add_argument("--every_n", type=int, default=1)
    ap.add_argument("--max_frames_per_hdf5", type=int, default=-1)

    # Keys (adjust if your hdf5 differs)
    ap.add_argument("--img_key", default="images/rgb_left")
    ap.add_argument("--gps_key", default="gps/latlong")
    ap.add_argument("--yaw_key", default="imu/compass_bearing")

    # Optional: local satellite mapping
    ap.add_argument("--sat_img", default="/data/tlxd/Cross-View/SCWM/tmp_test/rfs_satellite.png")
    ap.add_argument("--sat_nw_lat", type=float, default=37.915265)
    ap.add_argument("--sat_nw_lon", type=float, default=-122.334964)
    ap.add_argument("--sat_se_lat", type=float, default=37.914700)
    ap.add_argument("--sat_se_lon", type=float, default=-122.334358)

    args = ap.parse_args()

    # If satellite image provided, require corners
    if args.sat_img:
        if not (np.isfinite(args.sat_nw_lat) and np.isfinite(args.sat_nw_lon) and
                np.isfinite(args.sat_se_lat) and np.isfinite(args.sat_se_lon)):
            raise ValueError("sat_img provided but sat_nw_lat/lon and sat_se_lat/lon are not fully set.")

    extract_dataset(args)


if __name__ == "__main__":
    main()
