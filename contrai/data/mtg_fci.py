# contrai/data/mtg_fci.py
"""
Helpers for searching and downloading Meteosat Third Generation (MTG)
Flexible Combined Imager (FCI) Level 1c products from the EUMETSAT
Data Store using the EUMDAC client.

This module is library-only: no side effects, no printing.
"""

from __future__ import annotations

import io
import os
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence

import requests  # for URL fallback on EUMDAC 3.x
import eumdac  # type: ignore[import]


# Default collection ID for MTG FCI Level 1c High Resolution Fast Imagery (0°).
# Verify against EUMETSAT documentation for your use-case.
DEFAULT_FCI_L1C_COLLECTION_ID = "EO:EUM:DAT:0665"


def find_fci_l1c_products(
    datastore: eumdac.DataStore,
    start: datetime,
    end: datetime,
    *,
    collection_id: str = DEFAULT_FCI_L1C_COLLECTION_ID,
    bbox: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
) -> List[eumdac.dataset.Dataset]:
    """
    Search MTG FCI Level 1c products in the EUMETSAT Data Store.

    Parameters
    ----------
    datastore : eumdac.DataStore
        Authenticated DataStore instance. Authentication (API key/secret)
        must be handled by the caller, for example via
        :class:`eumdac.AccessToken`.
    start : datetime
        Start of search interval (UTC).
    end : datetime
        End of search interval (UTC). Must be strictly later than ``start``.
    collection_id : str, optional
        EUMETSAT collection identifier for the desired FCI L1c product.
        Defaults to ``"EO:EUM:DAT:0665"``.
    bbox : sequence of float, optional
        Geographic bounding box as ``[west, south, east, north]`` in degrees.
        If omitted, no spatial filter is applied.
    limit : int, optional
        Maximum number of products to return. If ``None``, all matches
        in the interval (subject to API defaults) are returned.

    Returns
    -------
    list of eumdac.dataset.Dataset
        List of matching product descriptors.

    Raises
    ------
    ValueError
        If ``start >= end`` or ``bbox`` is malformed.
    """
    if start >= end:
        raise ValueError("start must be earlier than end")

    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox must be [west, south, east, north]")

    collection = datastore.get_collection(collection_id)

    search_kwargs = {"dtstart": start, "dtend": end}
    if bbox is not None:
        search_kwargs["bbox"] = bbox

    products_iter = collection.search(**search_kwargs)

    if limit is None:
        return list(products_iter)

    products: List[eumdac.dataset.Dataset] = []
    for i, product in enumerate(products_iter):
        if i >= limit:
            break
        products.append(product)

    return products


# -------------------------------
# Download helpers (2.x and 3.x)
# -------------------------------

def _iter_eumdac_entries(product: Any) -> Iterable[Any]:
    """
    Yield entry-like objects for a product on EUMDAC 3.x.
    Tries common attributes: 'entries', 'assets', 'items'.
    Falls back to yielding the product itself if none exist.
    """
    for attr in ("entries", "assets", "items"):
        if hasattr(product, attr):
            coll = getattr(product, attr)
            try:
                for e in coll:
                    yield e
            except TypeError:
                # Not iterable; ignore and keep searching
                pass
            return
    # Fallback: treat the product itself as a single entry
    yield product


def _guess_entry_name(entry: Any) -> str:
    """Pick a reasonable filename for an entry."""
    if hasattr(entry, "filename") and getattr(entry, "filename"):
        return str(getattr(entry, "filename"))
    if hasattr(entry, "name") and getattr(entry, "name"):
        return str(getattr(entry, "name"))
    if hasattr(entry, "id") and getattr(entry, "id"):
        return str(getattr(entry, "id"))
    if isinstance(entry, dict):
        for k in ("filename", "name", "id"):
            v = entry.get(k)
            if v:
                return str(v)
    return "part"


def _extract_download_url(entry: Any) -> Optional[str]:
    """
    Extract a (pre-signed) HTTPS URL from an entry object that lacks
    .download()/.open()/.read(). Supports several common shapes.
    """
    # Direct attributes
    for attr in ("href", "url", "download_url", "downloadUri", "download_url_signed"):
        val = getattr(entry, attr, None)
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            return val

    # location.href pattern
    loc = getattr(entry, "location", None)
    if loc is not None:
        href = getattr(loc, "href", None)
        if isinstance(href, str) and href.startswith(("http://", "https://")):
            return href

    # links[...].href, rel in (enclosure/self/download)
    links = getattr(entry, "links", None) or (entry.get("links") if isinstance(entry, dict) else None)
    if links:
        for link in links:
            href = getattr(link, "href", None) or (link.get("href") if isinstance(link, dict) else None)
            rel = getattr(link, "rel", None) or (link.get("rel") if isinstance(link, dict) else None)
            if href and href.startswith(("http://", "https://")) and (rel in (None, "enclosure", "self", "download")):
                return href

    # dict-like direct URL fields
    if isinstance(entry, dict):
        for key in ("href", "url", "download", "downloadUri"):
            val = entry.get(key)
            if isinstance(val, str) and val.startswith(("http://", "https://")):
                return val

    return None


def _download_streamlike(obj: Any, target_path: Path) -> None:
    """
    Write bytes from a stream/response/bytes-like object into target_path.
    Supports .iter_content(), .read(), or raw bytes/bytearray.
    """
    with open(target_path, "wb") as fh:
        if hasattr(obj, "iter_content"):
            for chunk in obj.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
            return
        if hasattr(obj, "read"):
            fh.write(obj.read())
            return
        if isinstance(obj, (bytes, bytearray)):
            fh.write(obj)
            return
        content = getattr(obj, "content", None)
        if content is not None:
            fh.write(content)
            return
        raise RuntimeError("Unsupported stream object: cannot read bytes.")


def _entry_matches(entry: Any, allowed: Optional[Iterable[str]]) -> bool:
    """
    If 'allowed' is provided, only download entries whose id/name/filename
    matches one of the provided strings (exact match).
    """
    if not allowed:
        return True
    name = _guess_entry_name(entry)
    ids = {
        name,
        str(getattr(entry, "id", "")),
        str(getattr(entry, "name", "")),
        str(getattr(entry, "filename", "")),
    }
    return any(a in ids for a in allowed)


def _download_product(product: Any, out_dir: str, entries: Optional[Iterable[str]] = None) -> List[str]:
    """
    Download a product (EUMDAC v2) or its entries/chunks (EUMDAC 3).
    If 'entries' is provided, it's a whitelist of entry IDs/names to download.
    Returns list of file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- EUMDAC v2.x fast path: product has .download(...)
    if hasattr(product, "download"):
        try:
            p = product.download(out)
            return [str(p)] if p else []
        except TypeError:
            p = product.download(path=out)
            return [str(p)] if p else []

    # --- EUMDAC v3.x: iterate entry-like objects and fetch each
    downloaded: List[str] = []
    for entry in _iter_eumdac_entries(product):
        if not _entry_matches(entry, entries):
            continue

        name = _guess_entry_name(entry)
        target = out / name
        tmp_target = target.with_suffix(target.suffix + ".part")

        # Preferred: entry.download(...)
        if hasattr(entry, "download"):
            try:
                p = entry.download(out)
                downloaded.append(str(p) if p else str(out / name))
                continue
            except TypeError:
                p = entry.download(path=target)
                downloaded.append(str(p) if p else str(target))
                continue

        # Fallbacks: entry.open() / entry.read()
        if hasattr(entry, "open"):
            with entry.open() as stream:
                _download_streamlike(stream, tmp_target)
            os.replace(tmp_target, target)
            downloaded.append(str(target))
            continue

        if hasattr(entry, "read"):
            data = entry.read()
            _download_streamlike(io.BytesIO(data), tmp_target)
            os.replace(tmp_target, target)
            downloaded.append(str(target))
            continue

        # URL-based fallback (typical for some 3.x builds)
        url = _extract_download_url(entry)
        if url:
            with contextlib.ExitStack() as stack:
                resp = stack.enter_context(requests.get(url, stream=True, timeout=120))
                resp.raise_for_status()
                with open(tmp_target, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            os.replace(tmp_target, target)
            downloaded.append(str(target))
            continue

        # Last fallback: raw content attribute
        content = getattr(entry, "content", None) or (entry.get("content") if isinstance(entry, dict) else None)
        if isinstance(content, (bytes, bytearray)):
            _download_streamlike(content, tmp_target)
            os.replace(tmp_target, target)
            downloaded.append(str(target))
            continue

        raise RuntimeError("Unsupported EUMDAC v3 entry: no download/open/read/url/content.")

    return downloaded


def download_fci_l1c_products(
    datastore: eumdac.DataStore,
    start: datetime,
    end: datetime,
    *,
    out_dir: str,
    collection_id: str = DEFAULT_FCI_L1C_COLLECTION_ID,
    bbox: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
    entries: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Download MTG FCI Level 1c products for a given time range (and optional region).

    Parameters
    ----------
    datastore : eumdac.DataStore
        Authenticated DataStore instance.
    start : datetime
        Start of search interval (UTC).
    end : datetime
        End of search interval (UTC).
    out_dir : str
        Directory where products will be saved. Created if it does not exist.
    collection_id : str, optional
        FCI L1c collection identifier. Defaults to ``"EO:EUM:DAT:0665"``.
    bbox : sequence of float, optional
        Geographic bounding box as ``[west, south, east, north]`` in degrees.
    limit : int, optional
        Maximum number of products to download.
    entries : iterable of str, optional
        If provided, only download specific entries within each product.
        The meaning of entries is defined by the EUMETSAT product structure.

    Returns
    -------
    list of str
        Paths of downloaded files.
    """
    os.makedirs(out_dir, exist_ok=True)

    products = find_fci_l1c_products(
        datastore=datastore,
        start=start,
        end=end,
        collection_id=collection_id,
        bbox=bbox,
        limit=limit,
    )

    downloaded: List[str] = []
    for product in products:
        downloaded.extend(_download_product(product, out_dir, entries=entries))

    return downloaded


__all__ = [
    "DEFAULT_FCI_L1C_COLLECTION_ID",
    "find_fci_l1c_products",
    "download_fci_l1c_products",
]
