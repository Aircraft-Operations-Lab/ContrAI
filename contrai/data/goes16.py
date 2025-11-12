# contrai/data/goes16.py
"""
Utilities to locate and download GOES-16 ABI L1b radiance data
from the public NOAA S3 bucket, and generate GOES-16 Ash RGB imagery.

All functions are side-effect free except for explicit downloads
and optional PNG writing.
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

DEFAULT_BUCKET = "noaa-goes16"
DEFAULT_PRODUCT = "ABI-L1b-RadF"          # Full Disk L1b radiances
DEFAULT_ASH_CHANNELS = ("11", "13", "14", "15")
DEFAULT_MAX_TIME_DIFF = timedelta(minutes=30)

# Root directory to store downloaded L1b files
DEFAULT_OUT_ROOT = "images/goes16_l1b"

# Root directory to store generated Ash RGB PNGs
DEFAULT_ASH_RGB_ROOT = "images/goes16_ash_rgb"

# Default Ash RGB viewport and resolution
DEFAULT_ASH_LAT_BOUNDS = (15.0, 40.0)
DEFAULT_ASH_LON_BOUNDS = (-90.0, -30.0)
DEFAULT_ASH_RES_DEG = 0.02  # ~2 km

# Global debug flag for progress prints
DEBUG_GOES16 = True

SCAN_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def _log(message: str) -> None:
    """Lightweight debug logger for this module."""
    if DEBUG_GOES16:
        print(f"[GOES16] {message}")


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def target_datetime(year: int, month: int, day: int, hhmm: str) -> datetime:
    """
    Build a UTC datetime from components.
    """
    hour = int(hhmm[:2])
    minute = int(hhmm[2:4])
    dt = datetime(year, month, day, hour, minute)
    _log(f"Target datetime: {dt.isoformat()}Z")
    return dt


def day_of_year(dt: datetime) -> int:
    """
    Compute the day-of-year index for a given date.
    """
    doy = (dt - datetime(dt.year, 1, 1)).days + 1
    _log(f"Day-of-year for {dt.date()}: {doy:03d}")
    return doy


# -----------------------------------------------------------------------------
# Filename parsing
# -----------------------------------------------------------------------------

def extract_scan_time_from_name(name: str) -> Optional[datetime]:
    """
    Extract GOES-ABI scan start time embedded in a filename.

    Pattern: _sYYYYJJJHHMMSS_
    """
    m = SCAN_RE.search(name)
    if not m:
        return None
    year, jjj, hh, mm, ss = map(int, m.groups())
    dt0 = datetime(year, 1, 1) + timedelta(days=jjj - 1)
    return datetime(dt0.year, dt0.month, dt0.day, hh, mm, ss)


# -----------------------------------------------------------------------------
# Efficient S3 search
# -----------------------------------------------------------------------------

def _build_s3_filesystem() -> s3fs.S3FileSystem:
    """
    Construct an anonymous S3FileSystem with a slightly larger connection pool.
    """
    return s3fs.S3FileSystem(
        anon=True,
        config_kwargs={"max_pool_connections": 64},
    )


def _candidate_hour_prefixes(
    target: datetime,
    max_time_diff: Optional[timedelta],
    bucket: str,
    product: str,
) -> List[str]:
    """
    Build a list of S3 prefixes (hour directories) likely to contain scans
    near the target time.

    If max_time_diff is provided, restrict to hours overlapping the window
    [target - max_time_diff, target + max_time_diff].

    If max_time_diff is None, fall back to the entire UTC day of `target`.
    """
    prefixes: List[str] = []

    if max_time_diff is None:
        doy = day_of_year(target)
        # Whole day, hour subdirs will be discovered by fs.ls on this prefix.
        prefixes.append(f"{bucket}/{product}/{target.year:04d}/{doy:03d}/")
        return prefixes

    t_min = target - max_time_diff
    t_max = target + max_time_diff

    # Normalize to hour boundaries
    current = datetime(t_min.year, t_min.month, t_min.day, t_min.hour)
    end = datetime(t_max.year, t_max.month, t_max.day, t_max.hour)

    seen = set()
    step = timedelta(hours=1)

    while current <= end:
        doy = day_of_year(current)
        prefix = (
            f"{bucket}/{product}/"
            f"{current.year:04d}/{doy:03d}/{current.hour:02d}/"
        )
        if prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)
        current += step

    return prefixes

def find_ash_keys_for_datetime(
    year: int,
    month: int,
    day: int,
    hhmm: str,
    *,
    channels: Iterable[str] = DEFAULT_ASH_CHANNELS,
    bucket: str = DEFAULT_BUCKET,
    product: str = DEFAULT_PRODUCT,
    max_time_diff: Optional[timedelta] = DEFAULT_MAX_TIME_DIFF,
) -> Tuple[Dict[str, str], datetime]:
    """
    Locate ABI-L1b-RadF S3 object keys closest in time to the target.

    Optimized:
      - Only probe hour prefixes that could contain the target time.
      - Per-channel glob to avoid large directory listings.
      - Concurrency for prefix/channel discovery.
    """
    target = target_datetime(year, month, day, hhmm)
    fs = _build_s3_filesystem()

    prefixes = _candidate_hour_prefixes(target, max_time_diff, bucket, product)
    if not prefixes:
        raise FileNotFoundError("No candidate prefixes constructed for search window")

    if max_time_diff is not None:
        _log(
            f"Searching for Ash-RGB channels near {target.isoformat()}Z "
            f"within ±{max_time_diff}, across {len(prefixes)} hour-prefixes"
        )
    else:
        _log(
            f"Searching for Ash-RGB channels for {target.date()} "
            f"(no time window limit), base prefix count: {len(prefixes)}"
        )

    # Per-channel candidate lists using glob (fast server-side filtering)
    ch_to_candidates: Dict[str, List[str]] = {ch: [] for ch in channels}

    def _glob_one(prefix: str, ch: str) -> List[str]:
        """
        Try strict pattern first (correct for GOES-16): OR_<product>-M6C{ch}_*.nc
        Fall back to *<product>-M6C{ch}_*.nc in case of any unforeseen prefixing.
        """
        patterns = [
            f"{prefix}OR_{product}-M6C{ch}_*.nc",   # canonical
            f"{prefix}*{product}-M6C{ch}_*.nc",     # permissive fallback
        ]
        out: List[str] = []
        for pat in patterns:
            try:
                out.extend(fs.glob(pat))
            except FileNotFoundError:
                continue
        return out

    # Concurrency across (prefix, channel)
    max_workers = min(64, max(1, len(prefixes) * len(tuple(channels))))
    total_matches = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for prefix in prefixes:
            # ensure trailing slash for safety
            if not prefix.endswith("/"):
                prefix = prefix + "/"
            for ch in channels:
                futures[ex.submit(_glob_one, prefix, ch)] = ch

        for fut in as_completed(futures):
            ch = futures[fut]
            try:
                res = fut.result()
                ch_to_candidates[ch].extend(res)
                total_matches += len(res)
            except Exception as e:
                _log(f"Warning: glob failed for band {ch}: {e}")

    _log(f"Discovered {total_matches} candidate files across requested prefixes.")

    # Fallback for max_time_diff=None (whole-day search) if nothing came back
    if max_time_diff is None and all(len(v) == 0 for v in ch_to_candidates.values()):
        _log("Fallback: day-wide listing due to empty glob results")
        day_prefix = prefixes[0]  # .../YYYY/DOY/
        try:
            hour_dirs = fs.ls(day_prefix)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No data for {year:04d}-{month:02d}-{day:02d} "
                f"under s3://{bucket}/{product}/"
            ) from e

        hour_dirs = [p if p.endswith("/") else p + "/" for p in hour_dirs]
        with ThreadPoolExecutor(max_workers=32) as ex:
            futures = {}
            for hour_dir in hour_dirs:
                for ch in channels:
                    futures[ex.submit(_glob_one, hour_dir, ch)] = ch
            for fut in as_completed(futures):
                ch = futures[fut]
                try:
                    ch_to_candidates[ch].extend(fut.result())
                except Exception as e:
                    _log(f"Warning: day-glob failed for band {ch}: {e}")

    # Select best file per channel by time proximity
    keys: Dict[str, str] = {}
    for ch in channels:
        ch_files = ch_to_candidates.get(ch, [])
        if not ch_files:
            # Helpful hint for debugging if this ever happens again
            _log(f"No matches for pattern OR_{product}-M6C{ch}_*.nc "
                 f"around {target.isoformat()}Z in {len(prefixes)} prefixes.")
            raise FileNotFoundError(f"No files for band {ch} near {target.isoformat()}Z")

        best_file = None
        best_time = None
        best_diff: Optional[timedelta] = None

        for fpath in ch_files:
            t = extract_scan_time_from_name(os.path.basename(fpath))
            if t is None:
                continue
            diff = abs(t - target)
            if best_diff is None or diff < best_diff:
                best_file, best_time, best_diff = fpath, t, diff

        if best_file is None or best_time is None or best_diff is None:
            raise FileNotFoundError(
                f"No valid scan times for band {ch} near {target.isoformat()}Z"
            )

        if max_time_diff is not None and best_diff > max_time_diff:
            raise FileNotFoundError(
                "Closest file for band {ch} is at {bt} UTC, outside "
                "max_time_diff={mt} from target {tt} UTC. "
                "File: s3://{f}".format(
                    ch=ch,
                    bt=best_time.strftime("%Y-%m-%d %H:%M:%S"),
                    mt=str(max_time_diff),
                    tt=target.strftime("%Y-%m-%d %H:%M:%S"),
                    f=best_file,
                )
            )

        keys[ch] = best_file
        _log(
            f"Band {ch}: selected {os.path.basename(best_file)} "
            f"(scan start {best_time.isoformat()}Z, Δt={best_diff})"
        )

    return keys, target



# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

def download_ash_bands_for_datetime(
    year: int,
    month: int,
    day: int,
    hhmm: str,
    *,
    out_root: str = DEFAULT_OUT_ROOT,
    channels: Iterable[str] = DEFAULT_ASH_CHANNELS,
    bucket: str = DEFAULT_BUCKET,
    product: str = DEFAULT_PRODUCT,
    max_time_diff: Optional[timedelta] = DEFAULT_MAX_TIME_DIFF,
) -> Tuple[Dict[str, str], datetime]:
    """
    Download GOES-16 ABI-L1b-RadF Ash-RGB channel files for a given datetime.

    Returns
    -------
    (local_paths, target_dt)
        local_paths : dict
            Mapping {channel: local_path}
        target_dt : datetime
            Target datetime used for matching.
    """
    _log(
        f"Resolving and downloading Ash-RGB bands for "
        f"{year:04d}-{month:02d}-{day:02d} {hhmm} UTC"
    )

    keys, target = find_ash_keys_for_datetime(
        year=year,
        month=month,
        day=day,
        hhmm=hhmm,
        channels=channels,
        bucket=bucket,
        product=product,
        max_time_diff=max_time_diff,
    )

    fs = _build_s3_filesystem()

    subdir = os.path.join(
        out_root,
        f"{year:04d}",
        f"{month:02d}",
        f"{day:02d}",
        hhmm,
    )
    os.makedirs(subdir, exist_ok=True)
    _log(f"L1b output directory: {subdir}")

    local_paths: Dict[str, str] = {}
    for ch, key in keys.items():
        fname = os.path.basename(key)
        local_path = os.path.join(subdir, fname)
        if os.path.exists(local_path):
            _log(f"Overwriting existing file for band {ch}: {local_path}")
            os.remove(local_path)
        _log(f"Downloading band {ch} from s3://{key}")
        fs.get(key, local_path)
        local_paths[ch] = local_path

    _log("All requested bands downloaded successfully")
    return local_paths, target


# -----------------------------------------------------------------------------
# Ash RGB construction
# -----------------------------------------------------------------------------

def _load_bt(path: str):
    """
    Load brightness temperature and projection info from a GOES-16 ABI L1b file.

    Returns
    -------
    bt : np.ndarray (float32)
    proj : xarray.DataArray
    x, y : xarray.DataArray
    """
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


def _goes_latlon(proj, x, y):
    """
    Convert GOES fixed grid coordinates to lat/lon in degrees.
    """
    _log("Converting GOES fixed grid coordinates to lat/lon")
    Re = float(proj.semi_major_axis)
    Rp = float(proj.semi_minor_axis)
    H = float(proj.perspective_point_height) + Re
    lon0 = np.deg2rad(float(proj.longitude_of_projection_origin))

    xx, yy = np.meshgrid(x.values, y.values)

    sin_x = np.sin(xx)
    cos_x = np.cos(xx)
    sin_y = np.sin(yy)
    cos_y = np.cos(yy)

    a = (
        sin_x**2
        + cos_x**2 * (cos_y**2 + (Re**2 / Rp**2) * sin_y**2)
    )
    b = -2.0 * H * cos_x * cos_y
    c = H**2 - Re**2

    disc = b**2 - 4 * a * c
    mask = disc <= 0
    disc = np.where(mask, np.nan, disc)

    rs = (-b - np.sqrt(disc)) / (2 * a)

    Sx = rs * cos_x * cos_y
    Sy = rs * sin_x * cos_y
    Sz = rs * sin_y

    lat = np.arctan((Re**2 / Rp**2) * (Sz / np.sqrt((H - Sx) ** 2 + Sy**2)))
    lon = lon0 - np.arctan2(Sy, H - Sx)

    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)

    lat_deg[mask] = np.nan
    lon_deg[mask] = np.nan

    return lon_deg, lat_deg


def _clip_scale(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Linearly scale data to 0-1 and clip.
    """
    return np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)


def _build_area_def(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    res_deg: float,
) -> geometry.AreaDefinition:
    """
    Build a regular lat/lon AreaDefinition for pyresample.
    """
    _log(
        f"Creating target grid: "
        f"lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}], "
        f"res {res_deg} deg"
    )
    lats_tgt = np.arange(lat_min, lat_max + res_deg, res_deg)
    lons_tgt = np.arange(lon_min, lon_max + res_deg, res_deg)
    width, height = len(lons_tgt), len(lats_tgt)

    return geometry.AreaDefinition(
        "ash_latlon",
        "GOES-16 Ash RGB lat/lon grid",
        "latlon",
        {"proj": "longlat", "datum": "WGS84", "no_defs": True},
        width,
        height,
        (lon_min, lat_min, lon_max, lat_max),
    )


def _resample_channel(
    channel: np.ndarray,
    swath_def: geometry.SwathDefinition,
    area_def: geometry.AreaDefinition,
) -> np.ndarray:
    _log("Resampling channel to target grid")
    return kd_tree.resample_nearest(
        swath_def,
        channel,
        area_def,
        radius_of_influence=8000.0,  # ~1 ABI pixel
        fill_value=np.nan,
    )


def build_ash_rgb_from_paths(
    paths: Dict[str, str],
    *,
    lat_bounds: Tuple[float, float] = DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds: Tuple[float, float] = DEFAULT_ASH_LON_BOUNDS,
    res_deg: float = DEFAULT_ASH_RES_DEG,
) -> np.ndarray:
    """
    Construct GOES-16 Ash RGB composite from local L1b channel files.

    Parameters
    ----------
    paths : dict
        Mapping {"11": path_C11, "13": path_C13, "14": path_C14, "15": path_C15}.
    """
    _log("Building Ash RGB from local L1b files")

    for ch in ("11", "13", "14", "15"):
        if ch not in paths:
            raise KeyError(f"Missing required channel {ch!r} in paths")

    bt11, proj, x, y = _load_bt(paths["11"])
    bt13, _,   _, _  = _load_bt(paths["13"])
    bt14, _,   _, _  = _load_bt(paths["14"])
    bt15, _,   _, _  = _load_bt(paths["15"])

    lon_deg, lat_deg = _goes_latlon(proj, x, y)

    _log("Applying Ash RGB spectral differences and scaling")
    red = _clip_scale(bt15 - bt13, -4.0, 2.0)
    green = _clip_scale(bt14 - bt11, -4.0, 5.0)
    blue = _clip_scale(bt13, 243.0, 303.0)

    # Mask space
    space_mask = np.isnan(lat_deg) | np.isnan(lon_deg)
    red[space_mask] = np.nan
    green[space_mask] = np.nan
    blue[space_mask] = np.nan

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    area_def = _build_area_def(lat_min, lat_max, lon_min, lon_max, res_deg)
    swath_def = geometry.SwathDefinition(lons=lon_deg, lats=lat_deg)

    Rr = _resample_channel(red, swath_def, area_def)
    Gr = _resample_channel(green, swath_def, area_def)
    Br = _resample_channel(blue, swath_def, area_def)

    rgb = np.dstack(
        [
            np.nan_to_num(Rr, nan=0.0),
            np.nan_to_num(Gr, nan=0.0),
            np.nan_to_num(Br, nan=0.0),
        ]
    )
    rgb = np.clip(rgb, 0.0, 1.0)

    _log("Ash RGB array constructed successfully")
    return rgb


def generate_ash_rgb_for_datetime(
    year: int,
    month: int,
    day: int,
    hhmm: str,
    *,
    l1b_root: str = DEFAULT_OUT_ROOT,
    rgb_root: str = DEFAULT_ASH_RGB_ROOT,
    lat_bounds: Tuple[float, float] = DEFAULT_ASH_LAT_BOUNDS,
    lon_bounds: Tuple[float, float] = DEFAULT_ASH_LON_BOUNDS,
    res_deg: float = DEFAULT_ASH_RES_DEG,
    save_png: bool = True,
) -> Tuple[str, np.ndarray, datetime]:
    """
    High-level helper:
    - Download GOES-16 L1b Ash-RGB channels for a datetime.
    - Generate Ash RGB.
    - Save PNG under a default path (if requested).

    PNG path layout:
        {rgb_root}/{YYYY}/{MM}/{DD}/{HHMM}/ash_rgb_{res}deg.png
    """
    _log(
        f"Starting Ash RGB generation for "
        f"{year:04d}-{month:02d}-{day:02d} {hhmm} UTC"
    )

    local_paths, target_dt = download_ash_bands_for_datetime(
        year,
        month,
        day,
        hhmm,
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
    out_dir = os.path.join(
        rgb_root,
        f"{year:04d}",
        f"{month:02d}",
        f"{day:02d}",
        hhmm,
    )
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"ash_rgb_{res_str}deg.png")

    if save_png:
        _log(f"Saving Ash RGB PNG to {png_path}")
        plt.imsave(png_path, rgb)
    else:
        _log("PNG saving disabled; returning RGB array only")

    _log(
        f"Ash RGB generation complete "
        f"(scan time {target_dt.isoformat()}Z)"
    )
    return png_path, rgb, target_dt


__all__ = [
    "target_datetime",
    "day_of_year",
    "extract_scan_time_from_name",
    "find_ash_keys_for_datetime",
    "download_ash_bands_for_datetime",
    "build_ash_rgb_from_paths",
    "generate_ash_rgb_for_datetime",
]
