"""
Utilities to locate and download GOES-16 ABI L1b radiance data
from the public NOAA S3 bucket, and generate GOES-16 Ash RGB imagery.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Iterable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import s3fs
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyresample import geometry, kd_tree

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_TRUE_COLOR_CHANNELS = ("01", "02", "03")
DEFAULT_TRUE_L1B_ROOT = "images/goes_l1b"
DEFAULT_TRUE_RGB_ROOT = "images/goes16_true_rgb"

DEFAULT_BUCKET = "noaa-goes16"
DEFAULT_PRODUCT = "ABI-L1b-RadF"
DEFAULT_ASH_CHANNELS = ("11", "13", "14", "15")
DEFAULT_MAX_TIME_DIFF = timedelta(minutes=30)

DEFAULT_OUT_ROOT = "images/goes16_l1b"
DEFAULT_ASH_RGB_ROOT = "images/goes16_ash_rgb"

DEFAULT_ASH_LAT_BOUNDS = (-60.0, 40.0)
DEFAULT_ASH_LON_BOUNDS = (-135.0, -15.0)
DEFAULT_ASH_RES_DEG = 0.02  # ~2 km

DEBUG_GOES16 = True
SCAN_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")

# Global cache for S3 filesystem
_S3_FS = None


def _log(msg: str):
    if DEBUG_GOES16:
        print(f"[GOES16] {msg}")


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def target_datetime(year: int, month: int, day: int, hhmm: str) -> datetime:
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    dt = datetime(year, month, day, hour, minute)
    _log(f"Target datetime: {dt.isoformat()}Z")
    return dt


def day_of_year(dt: datetime) -> int:
    return (dt - datetime(dt.year, 1, 1)).days + 1


# -----------------------------------------------------------------------------
# Filename parsing
# -----------------------------------------------------------------------------

def extract_scan_time_from_name(name: str) -> Optional[datetime]:
    m = SCAN_RE.search(name)
    if not m:
        return None
    year, jjj, hh, mm, ss = map(int, m.groups())
    base = datetime(year, 1, 1) + timedelta(days=jjj - 1)
    return datetime(base.year, base.month, base.day, hh, mm, ss)


# -----------------------------------------------------------------------------
# S3 helpers
# -----------------------------------------------------------------------------

def _build_s3_filesystem():
    """
    Build or reuse a global anonymous S3 filesystem with a larger connection pool.
    """
    global _S3_FS
    if _S3_FS is None:
        _S3_FS = s3fs.S3FileSystem(
            anon=True,
            config_kwargs={"max_pool_connections": 128},
            default_fill_cache=False,
        )
        _log("Initialized global S3 filesystem")
    return _S3_FS


def _candidate_hour_prefixes(target, max_time_diff, bucket, product):
    if max_time_diff is None:
        doy = day_of_year(target)
        return [f"{bucket}/{product}/{target.year:04d}/{doy:03d}/"]

    tmin = target - max_time_diff
    tmax = target + max_time_diff

    prefixes = []
    current = datetime(tmin.year, tmin.month, tmin.day, tmin.hour)
    end = datetime(tmax.year, tmax.month, tmax.day, tmax.hour)

    seen = set()
    while current <= end:
        doy = day_of_year(current)
        prefix = f"{bucket}/{product}/{current.year:04d}/{doy:03d}/{current.hour:02d}/"
        if prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)
        current += timedelta(hours=1)
    return prefixes


def _glob_one(fs, prefix: str, ch: str, product: str) -> List[str]:
    """
    Find candidate keys for a given prefix/channel.

    Try the canonical pattern first; only fall back to a loose pattern if needed.
    """
    out: List[str] = []

    if not prefix.endswith("/"):
        prefix = prefix + "/"

    # Canonical GOES-16 naming pattern
    pat_main = f"{prefix}OR_{product}-M6C{ch}_*.nc"
    try:
        out = fs.glob(pat_main)
        if out:
            return out
    except FileNotFoundError:
        pass

    # Fallback, slightly looser pattern
    pat_fallback = f"{prefix}*{product}-M6C{ch}_*.nc"
    try:
        out = fs.glob(pat_fallback)
    except FileNotFoundError:
        out = []

    return out


# -----------------------------------------------------------------------------
# Find matching L1b files
# -----------------------------------------------------------------------------

def find_ash_keys_for_datetime(
    year, month, day, hhmm,
    *,
    channels=DEFAULT_ASH_CHANNELS,
    bucket=DEFAULT_BUCKET,
    product=DEFAULT_PRODUCT,
    max_time_diff=DEFAULT_MAX_TIME_DIFF,
):
    target = target_datetime(year, month, day, hhmm)
    fs = _build_s3_filesystem()
    prefixes = _candidate_hour_prefixes(target, max_time_diff, bucket, product)

    ch_to_candidates: Dict[str, List[str]] = {ch: [] for ch in channels}

    # Parallel globbing across prefixes *and* channels
    max_workers = min(32, max(1, len(prefixes) * len(channels)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_glob_one, fs, p, ch, product): ch
            for p in prefixes for ch in channels
        }

        for fut in as_completed(futures):
            ch = futures[fut]
            try:
                ch_to_candidates[ch].extend(fut.result())
            except Exception as e:
                _log(f"Glob failed for band {ch}: {e}")

    keys: Dict[str, str] = {}
    for ch, candidates in ch_to_candidates.items():
        if not candidates:
            raise FileNotFoundError(f"No files found for band {ch}")

        best_file = None
        best_diff = None
        best_time = None

        for fpath in candidates:
            t = extract_scan_time_from_name(os.path.basename(fpath))
            if t is None:
                continue
            diff = abs(t - target)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_file = fpath
                best_time = t

        if best_file is None:
            raise FileNotFoundError(f"No valid file for band {ch}")

        keys[ch] = best_file
        _log(f"Band {ch}: using file {os.path.basename(best_file)} (Δt={best_diff})")

    return keys, target


# -----------------------------------------------------------------------------
# Download L1b files (always used)
# -----------------------------------------------------------------------------

def _download_one_band(fs, ch: str, key: str, local_path: str):
    """
    Helper for parallel band download.
    """
    if os.path.exists(local_path):
        _log(f"Using cached file for band {ch}: {local_path}")
        return ch, local_path

    _log(f"Downloading band {ch} → {local_path}")
    fs.get(key, local_path)
    return ch, local_path


def download_ash_bands_for_datetime(
    year, month, day, hhmm,
    *,
    out_root=DEFAULT_OUT_ROOT,
    channels=DEFAULT_ASH_CHANNELS,
    bucket=DEFAULT_BUCKET,
    product=DEFAULT_PRODUCT,
    max_time_diff=DEFAULT_MAX_TIME_DIFF,
):
    keys, target_dt = find_ash_keys_for_datetime(
        year, month, day, hhmm,
        channels=channels,
        bucket=bucket,
        product=product,
        max_time_diff=max_time_diff,
    )

    fs = _build_s3_filesystem()

    out_dir = os.path.join(out_root, f"{year:04d}", f"{month:02d}", f"{day:02d}", hhmm)
    os.makedirs(out_dir, exist_ok=True)

    local_paths: Dict[str, str] = {}

    # Parallel downloads for all bands
    with ThreadPoolExecutor(max_workers=len(keys)) as ex:
        futures = []
        for ch, key in keys.items():
            fname = os.path.basename(key)
            local_path = os.path.join(out_dir, fname)
            futures.append(ex.submit(_download_one_band, fs, ch, key, local_path))

        for fut in as_completed(futures):
            ch, local_path = fut.result()
            local_paths[ch] = local_path

    return local_paths, target_dt


# -----------------------------------------------------------------------------
# Load L1b radiances (LOCAL-ONLY)
# -----------------------------------------------------------------------------

def _load_bt(path: str):
    _log(f"Loading brightness temperature from {path}")

    with xr.open_dataset(path) as ds:
        rad = ds["Rad"]
        if "t" in rad.dims:
            rad = rad.isel(t=0)
        rad = rad.values

        fk1 = float(ds["planck_fk1"])
        fk2 = float(ds["planck_fk2"])
        bc1 = float(ds["planck_bc1"])
        bc2 = float(ds["planck_bc2"])

        bt = fk2 / np.log(fk1 / (rad + bc1) + 1.0) + bc2

        proj = ds["goes_imager_projection"]
        x = ds["x"]
        y = ds["y"]

    return bt.astype(np.float32), proj, x, y


def _load_reflectance(path: str):
    """
    Load visible-band reflectance from a GOES-16 ABI L1b file.

    Uses kappa0 if available: reflectance = Rad * kappa0.
    """
    _log(f"Loading reflectance from {path}")

    with xr.open_dataset(path) as ds:
        rad = ds["Rad"]
        if "t" in rad.dims:
            rad = rad.isel(t=0)

        rad = rad.values.astype(np.float32)

        # Convert to reflectance if kappa0 is provided (GOES-16 VIS bands)
        if "kappa0" in ds:
            kappa0 = float(ds["kappa0"])
            ref = rad * kappa0
        else:
            # Fallback: keep as radiance, but this is less ideal
            ref = rad

        proj = ds["goes_imager_projection"]
        x = ds["x"]
        y = ds["y"]

    return ref, proj, x, y


# -----------------------------------------------------------------------------
# RGB construction
# -----------------------------------------------------------------------------

def _goes_latlon(proj, x, y):
    _log("Converting GOES fixed grid to lat/lon")
    Re = float(proj.semi_major_axis)
    Rp = float(proj.semi_minor_axis)
    H = float(proj.perspective_point_height) + Re
    lon0 = np.deg2rad(float(proj.longitude_of_projection_origin))

    xx, yy = np.meshgrid(x.values, y.values)

    sinx = np.sin(xx)
    cosx = np.cos(xx)
    siny = np.sin(yy)
    cosy = np.cos(yy)

    a = sinx**2 + cosx**2 * (cosy**2 + (Re**2 / Rp**2) * siny**2)
    b = -2 * H * cosx * cosy
    c = H**2 - Re**2
    disc = b**2 - 4 * a * c

    mask = disc <= 0
    disc = np.where(mask, np.nan, disc)

    rs = (-b - np.sqrt(disc)) / (2 * a)

    Sx = rs * cosx * cosy
    Sy = rs * sinx * cosy
    Sz = rs * siny

    lat = np.arctan((Re**2 / Rp**2) * (Sz / np.sqrt((H - Sx)**2 + Sy**2)))
    lon = lon0 - np.arctan2(Sy, H - Sx)

    latdeg = np.rad2deg(lat)
    londeg = np.rad2deg(lon)

    latdeg[mask] = np.nan
    londeg[mask] = np.nan
    return londeg, latdeg


def _clip_scale(arr, vmin, vmax):
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


def _build_area_def(latmin, latmax, lonmin, lonmax, res):
    _log("Building target grid")
    lats = np.arange(latmin, latmax + res, res)
    lons = np.arange(lonmin, lonmax + res, res)
    width, height = len(lons), len(lats)

    return geometry.AreaDefinition(
        "ash_latlon",
        "GOES-16 Ash RGB",
        "latlon",
        {"proj": "longlat", "datum": "WGS84"},
        width,
        height,
        (lonmin, latmin, lonmax, latmax),
    )


def _resample_channel(channel, swath, area):
    return kd_tree.resample_nearest(
        swath, channel, area,
        radius_of_influence=8000,
        fill_value=np.nan
    )


def build_ash_rgb_from_paths(
    paths, *,
    lat_bounds=DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds=DEFAULT_ASH_LON_BOUNDS,
    res_deg=DEFAULT_ASH_RES_DEG,
):
    bt11, proj, x, y = _load_bt(paths["11"])
    bt13, _, _, _ = _load_bt(paths["13"])
    bt14, _, _, _ = _load_bt(paths["14"])
    bt15, _, _, _ = _load_bt(paths["15"])

    lon, lat = _goes_latlon(proj, x, y)

    red = _clip_scale(bt15 - bt13, -4, 2)
    green = _clip_scale(bt14 - bt11, -4, 5)
    blue = _clip_scale(bt13, 243, 303)

    mask = np.isnan(lat) | np.isnan(lon)
    red[mask] = np.nan
    green[mask] = np.nan
    blue[mask] = np.nan

    latmin, latmax = lat_bounds
    lonmin, lonmax = lon_bounds

    area = _build_area_def(latmin, latmax, lonmin, lonmax, res_deg)
    swath = geometry.SwathDefinition(lons=lon, lats=lat)

    Rr = _resample_channel(red, swath, area)
    Gr = _resample_channel(green, swath, area)
    Br = _resample_channel(blue, swath, area)

    rgb = np.dstack([
        np.nan_to_num(Rr),
        np.nan_to_num(Gr),
        np.nan_to_num(Br)
    ])
    rgb = np.flip(rgb, axis=1)
    return np.clip(rgb, 0, 1)


def _downsample_to_match(high_res: np.ndarray, low_res: np.ndarray) -> np.ndarray:
    """
    Downsample high_res to the shape of low_res by block-averaging.

    Assumes:
      high_res.shape is an integer multiple of low_res.shape
      (for GOES-16 VIS, it's typically exactly 2x in each dimension).
    """
    hy, hx = high_res.shape
    ly, lx = low_res.shape

    fy = hy // ly
    fx = hx // lx

    # Trim any extra pixels that don't fit evenly (just in case)
    high_res = high_res[:ly * fy, :lx * fx]

    # Reshape and average blocks
    high_res = high_res.reshape(ly, fy, lx, fx).mean(axis=(1, 3))
    return high_res


# -----------------------------------------------------------------------------
# NEW: Geolocations of each pixel on the Ash RGB grid
# -----------------------------------------------------------------------------

def get_ash_rgb_pixel_geolocations(
    lat_bounds: Tuple[float, float] = DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds: Tuple[float, float] = DEFAULT_ASH_LON_BOUNDS,
    res_deg: float = DEFAULT_ASH_RES_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the latitude/longitude of the *center* of each Ash RGB pixel.

    Parameters
    ----------
    lat_bounds : (float, float)
        (lat_min, lat_max) used for the Ash RGB grid.
    lon_bounds : (float, float)
        (lon_min, lon_max) used for the Ash RGB grid.
    res_deg : float
        Grid resolution in degrees, same as used in _build_area_def.

    Notes
    -----
    - Must match the values used when generating the Ash RGB:
        build_ash_rgb_from_paths(..., lat_bounds=..., lon_bounds=..., res_deg=...)
    - Images are assumed plotted with:
        extent = [lon_min, lon_max, lat_min, lat_max]

    Returns
    -------
    lat_grid : (H, W) array of float
    lon_grid : (H, W) array of float
        These shapes should match rgb.shape[:2].
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    res = res_deg

    # Reproduce the target grid size used in _build_area_def
    lats_edge = np.arange(lat_min, lat_max + res, res)
    lons_edge = np.arange(lon_min, lon_max + res, res)
    height = len(lats_edge)     # number of rows (y)
    width = len(lons_edge)      # number of cols (x)

    # Given Matplotlib's imshow + extent semantics and pyresample's AreaDefinition,
    # the pixel centers are offset half a pixel inside the outer extent.
    dlat = (lat_max - lat_min) / height
    dlon = (lon_max - lon_min) / width

    lat_centers = lat_min + (np.arange(height) + 0.5) * dlat
    lon_centers = lon_min + (np.arange(width) + 0.5) * dlon

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    return lat_grid, lon_grid


def _to_true_color(red: np.ndarray,
                   veggie: np.ndarray,
                   blue: np.ndarray,
                   gamma: float = 2.2) -> np.ndarray:
    """
    Apply the Unidata-style true color recipe:
      - clip+scale each band [0,1]
      - gamma correction
      - synthetic green from red/veggie/blue
    """
    red = _clip_scale(red, 0.0, 1.0)
    veggie = _clip_scale(veggie, 0.0, 1.0)
    blue = _clip_scale(blue, 0.0, 1.0)

    red = red ** (1.0 / gamma)
    veggie = veggie ** (1.0 / gamma)
    blue = blue ** (1.0 / gamma)

    green = 0.45 * red + 0.1 * veggie + 0.45 * blue
    green = _clip_scale(green, 0.0, 1.0)

    rgb = np.dstack([red, green, blue]).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def build_truecolor_rgb_from_paths(
    paths,
    *,
    lat_bounds=DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds=DEFAULT_ASH_LON_BOUNDS,
    res_deg=DEFAULT_ASH_RES_DEG,
    gamma: float = 2.2,
):
    """
    Build a true-color RGB image from GOES-16 ABI L1b visible bands.

    Expects:
      paths["01"], paths["02"], paths["03"] → C01, C02, C03

    Returns
    -------
    rgb : (H, W, 3) array in [0, 1]
    """
    # Load reflectances
    c01, proj, x, y = _load_reflectance(paths["01"])  # blue
    c02, _, _, _ = _load_reflectance(paths["02"])      # red (hi-res)
    c03, _, _, _ = _load_reflectance(paths["03"])      # veggie

    # Downsample C02 (0.5 km) to 1 km grid if needed
    if c02.shape != c01.shape:
        c02 = _downsample_to_match(c02, c01)

    # Fixed-grid to lat/lon using the 1 km grid
    lon, lat = _goes_latlon(proj, x, y)

    # Mask invalid locations
    bad = np.isnan(lat) | np.isnan(lon)
    c01[bad] = np.nan
    c02[bad] = np.nan
    c03[bad] = np.nan

    latmin, latmax = lat_bounds
    lonmin, lonmax = lon_bounds

    area = _build_area_def(latmin, latmax, lonmin, lonmax, res_deg)
    swath = geometry.SwathDefinition(lons=lon, lats=lat)

    # Resample each band to target lat/lon grid
    R1 = _resample_channel(c01, swath, area)  # blue
    R2 = _resample_channel(c02, swath, area)  # red
    R3 = _resample_channel(c03, swath, area)  # veggie

    # NaNs → 0 before true-color math
    R1 = np.nan_to_num(R1)
    R2 = np.nan_to_num(R2)
    R3 = np.nan_to_num(R3)

    # Apply your true-color recipe
    rgb = _to_true_color(
        red=R2,
        veggie=R3,
        blue=R1,
        gamma=gamma,
    )
    rgb = np.flip(rgb, axis=1)
    return rgb


# -----------------------------------------------------------------------------
# High-level function (ALWAYS downloads)
# -----------------------------------------------------------------------------

def generate_ash_rgb_for_datetime(
    year, month, day, hhmm,
    *,
    l1b_root=DEFAULT_OUT_ROOT,
    rgb_root=DEFAULT_ASH_RGB_ROOT,
    lat_bounds=DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds=DEFAULT_ASH_LON_BOUNDS,
    res_deg=DEFAULT_ASH_RES_DEG,
    save_png=True,
):

    _log(f"Generating Ash RGB for {year}-{month}-{day} {hhmm}Z")

    local_paths, target_dt = download_ash_bands_for_datetime(
        year, month, day, hhmm,
        out_root=l1b_root,
        channels=DEFAULT_ASH_CHANNELS,
        bucket=DEFAULT_BUCKET,
        product=DEFAULT_PRODUCT,
        max_time_diff=DEFAULT_MAX_TIME_DIFF,
    )

    rgb = build_ash_rgb_from_paths(
        local_paths,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        res_deg=res_deg,
    )

    res_str = str(res_deg).replace(".", "p")
    out_dir = os.path.join(rgb_root, f"{year:04d}", f"{month:02d}", f"{day:02d}", hhmm)
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"ash_rgb_{res_str}deg.png")

    if save_png:
        plt.imsave(png_path, rgb)

    return png_path, rgb, target_dt


def generate_truecolor_rgb_for_datetime(
    year,
    month,
    day,
    hhmm,
    *,
    l1b_root=DEFAULT_TRUE_L1B_ROOT,
    rgb_root=DEFAULT_TRUE_RGB_ROOT,
    lat_bounds=DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds=DEFAULT_ASH_LON_BOUNDS,
    res_deg=DEFAULT_ASH_RES_DEG,
    save_png=True,
):
    """
    High-level helper:

    - Downloads the GOES-16 visible bands needed for a true-color composite
      (C01, C02, C03) into l1b_root (default: goes_l1b).
    - Builds the true-color RGB on the same lat/lon grid as Ash RGB.
    - Saves the PNG into rgb_root (default: goes16_true_rgb).

    Returns
    -------
    png_path : str
        Path to the saved PNG (if save_png=True)
    rgb : np.ndarray
        True-color RGB array (H, W, 3) in [0, 1]
    target_dt : datetime
        The target datetime used for matching scans.
    """
    _log(f"Generating True-Color RGB for {year}-{month}-{day} {hhmm}Z")

    # Reuse the generic downloader, but request true-color channels
    local_paths, target_dt = download_ash_bands_for_datetime(
        year,
        month,
        day,
        hhmm,
        out_root=l1b_root,
        channels=DEFAULT_TRUE_COLOR_CHANNELS,
        bucket=DEFAULT_BUCKET,
        product=DEFAULT_PRODUCT,
        max_time_diff=DEFAULT_MAX_TIME_DIFF,
    )

    rgb = build_truecolor_rgb_from_paths(
        local_paths,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        res_deg=res_deg,
    )

    res_str = str(res_deg).replace(".", "p")
    out_dir = os.path.join(
        rgb_root,
        f"{year:04d}",
        f"{month:02d}",
        f"{day:02d}",
        hhmm,
    )
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"true_rgb_{res_str}deg.png")

    if save_png:
        plt.imsave(png_path, rgb)

    return png_path, rgb, target_dt


__all__ = [
    "target_datetime",
    "day_of_year",
    "extract_scan_time_from_name",
    "find_ash_keys_for_datetime",
    "download_ash_bands_for_datetime",
    "build_ash_rgb_from_paths",
    "generate_ash_rgb_for_datetime",
    "get_ash_rgb_pixel_geolocations",
    "build_truecolor_rgb_from_paths",
    "generate_truecolor_rgb_for_datetime",
]
