import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import s3fs

# ========== CONFIG ==========

BUCKET = "noaa-goes16"
PRODUCT = "ABI-L1b-RadF"          # Full Disk L1b radiances
ASH_CHANNELS = ["11", "13", "14", "15"]  # Bands needed for Ash RGB

# Select target time (UTC)
YEAR = 2025
MONTH = 1
DAY = 5
TIME_HHMM = "1030"               # "HHMM"

# Maximum allowed time difference between requested time and scan start
MAX_TIME_DIFF = timedelta(minutes=30)     # set to None to disable

# Local output root for downloads
OUT_ROOT = "goes16_ash_l1b"

# ========== HELPERS ==========

SCAN_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def target_datetime(year: int, month: int, day: int, hhmm: str) -> datetime:
    hour = int(hhmm[:2])
    minute = int(hhmm[2:4])
    return datetime(year, month, day, hour, minute, 0)


def day_of_year(dt: datetime) -> int:
    return (dt - datetime(dt.year, 1, 1)).days + 1


def extract_scan_time_from_name(name: str) -> Optional[datetime]:
    """
    Extract GOES ABI scan start time from key or filename.
    Pattern: _sYYYYJJJHHMMSS_
    """
    m = SCAN_RE.search(name)
    if not m:
        return None
    year, jjj, hh, mm, ss = map(int, m.groups())
    dt0 = datetime(year, 1, 1) + timedelta(days=jjj - 1)
    return datetime(dt0.year, dt0.month, dt0.day, hh, mm, ss)


# ========== CORE LOGIC ==========

def find_ash_keys_for_datetime(
    year: int,
    month: int,
    day: int,
    hhmm: str,
    channels=ASH_CHANNELS,
    max_time_diff: Optional[timedelta] = MAX_TIME_DIFF,
) -> Tuple[Dict[str, str], datetime]:
    """
    For the requested UTC datetime, find for each Ash-RGB channel
    (C11, C13, C14, C15) the ABI-L1b-RadF file ON THE SAME DAY whose scan
    start time is closest to the target.

    Only returns success if:
      - All channels are found for that day.
      - Each chosen file is within max_time_diff of target (if max_time_diff not None).

    Returns:
      keys:   {channel: s3_key}
      target: datetime of requested target
    """
    fs = s3fs.S3FileSystem(anon=True)

    target = target_datetime(year, month, day, hhmm)
    doy = day_of_year(target)

    day_prefix = f"{BUCKET}/{PRODUCT}/{year:04d}/{doy:03d}/"
    print(f"Looking in s3://{day_prefix} for ~{target:%Y-%m-%d %H:%M} UTC")

    try:
        hour_dirs = fs.ls(day_prefix)
    except FileNotFoundError:
        raise FileNotFoundError(
            "No data for {y}-{m:02d}-{d:02d} (doy {doy:03d}) under s3://{b}/{p}/".format(
                y=year, m=month, d=day, doy=doy, b=BUCKET, p=PRODUCT
            )
        )

    all_files = []
    for hour_dir in hour_dirs:
        try:
            files = fs.ls(hour_dir)
        except FileNotFoundError:
            continue
        all_files.extend(files)

    if not all_files:
        raise FileNotFoundError(
            "No files found for {y}-{m:02d}-{d:02d} (doy {doy:03d}) under s3://{b}/{p}/".format(
                y=year, m=month, d=day, doy=doy, b=BUCKET, p=PRODUCT
            )
        )

    keys: Dict[str, str] = {}

    for ch in channels:
        ch_files = [
            f for f in all_files
            if ("%s-M6C%s_" % (PRODUCT, ch)) in f and f.endswith(".nc")
        ]
        if not ch_files:
            raise FileNotFoundError(
                "No files for band {ch} on {y}-{m:02d}-{d:02d}".format(
                    ch=ch, y=year, m=month, d=day
                )
            )

        best_file = None
        best_time = None
        best_diff = None

        for fpath in ch_files:
            t = extract_scan_time_from_name(os.path.basename(fpath))
            if t is None:
                continue
            if t.date() != target.date():
                continue

            diff = abs(t - target)
            if (best_diff is None) or (diff < best_diff):
                best_file = fpath
                best_time = t
                best_diff = diff

        if best_file is None or best_time is None or best_diff is None:
            raise FileNotFoundError(
                "No valid scan times for band {ch} on {y}-{m:02d}-{d:02d}".format(
                    ch=ch, y=year, m=month, d=day
                )
            )

        if (max_time_diff is not None) and (best_diff > max_time_diff):
            raise FileNotFoundError(
                "Closest file for band {ch} is at {bt} UTC, "
                "outside max_time_diff={mt} from target {tt} UTC.\n"
                "File: s3://{f}".format(
                    ch=ch,
                    bt=best_time.strftime("%Y-%m-%d %H:%M:%S"),
                    mt=str(max_time_diff),
                    tt=target.strftime("%Y-%m-%d %H:%M:%S"),
                    f=best_file,
                )
            )

        print(
            "Band {ch}: using s3://{f} (scan {bt} UTC, Δt={dt})".format(
                ch=ch,
                f=best_file,
                bt=best_time.strftime("%Y-%m-%d %H:%M:%S"),
                dt=str(best_diff),
            )
        )
        keys[ch] = best_file

    return keys, target


def download_ash_bands_for_datetime(
    year: int,
    month: int,
    day: int,
    hhmm: str,
    out_root: str = OUT_ROOT,
    channels=ASH_CHANNELS,
    max_time_diff: Optional[timedelta] = MAX_TIME_DIFF,
) -> Tuple[Dict[str, str], datetime]:
    """
    Find and download Ash-RGB ABI-L1b-RadF bands for the given datetime.

    Files are saved under:
        out_root/YYYY/MM/DD/HHMM/<original_filename>.nc

    Returns:
        local_paths: {channel: local_path}
        target:      datetime of the requested target
    """
    fs = s3fs.S3FileSystem(anon=True)

    keys, target = find_ash_keys_for_datetime(
        year, month, day, hhmm,
        channels=channels,
        max_time_diff=max_time_diff,
    )

    subdir = os.path.join(
        out_root,
        "{:04d}".format(year),
        "{:02d}".format(month),
        "{:02d}".format(day),
        hhmm,
    )
    os.makedirs(subdir, exist_ok=True)

    print("\nDownloading files into: {sd}".format(sd=subdir))

    local_paths: Dict[str, str] = {}

    for ch, key in keys.items():
        fname = os.path.basename(key)
        local_path = os.path.join(subdir, fname)

        print("  s3://{k} -> {lp}".format(k=key, lp=local_path))
        if os.path.exists(local_path):
            os.remove(local_path)
        fs.get(key, local_path)

        local_paths[ch] = local_path

    return local_paths, target


# ========== MAIN ==========

if __name__ == "__main__":
    req = target_datetime(YEAR, MONTH, DAY, TIME_HHMM)
    print("Requested GOES-16 Ash-RGB time: {t} UTC\n".format(
        t=req.strftime("%Y-%m-%d %H:%M")
    ))

    local_paths, used_time = download_ash_bands_for_datetime(
        YEAR, MONTH, DAY, TIME_HHMM
    )

    print("\nDownloaded Ash RGB bands:")
    for ch, path in local_paths.items():
        print("  Band {ch}: {p}".format(ch=ch, p=path))

    print("\nRequested time: {t} UTC".format(
        t=req.strftime("%Y-%m-%d %H:%M:%S")
    ))
    print("Download completed at: {t} UTC".format(
        t=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    ))
