# contrai/data/mtg_fci.py
"""
Helpers for searching and downloading Meteosat Third Generation (MTG)
Flexible Combined Imager (FCI) Level 1c products from the EUMETSAT
Data Store using the EUMDAC client.

This module is library-only: no side effects, no printing.

Notes
-----
The current implementation expects an ``eumdac`` version whose
product objects expose a :meth:`download` method. If your environment
uses a newer, incompatible ``eumdac`` API, a :class:`RuntimeError`
will be raised with guidance.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Union

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

    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be [west, south, east, north]")

    collection = datastore.get_collection(collection_id)

    search_kwargs = {
        "dtstart": start,
        "dtend": end,
    }
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


def _download_product(
    product: object,
    out_dir: str,
    entries: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Internal helper to download a single product using the installed eumdac API.

    Parameters
    ----------
    product : object
        Product-like object returned by ``collection.search``.
    out_dir : str
        Output directory for downloaded data.
    entries : iterable of str, optional
        Optional subset of entries/files to download.

    Returns
    -------
    list of str
        Paths to downloaded files.

    Raises
    ------
    RuntimeError
        If the installed ``eumdac`` version does not provide a compatible
        ``download`` API on the product object.
    """
    # EUMDAC classic API: Dataset.download(...)
    if hasattr(product, "download"):
        kwargs = {}
        if entries is not None:
            kwargs["entries"] = list(entries)
        result = product.download(out_dir, **kwargs)  # type: ignore[call-arg]

        # Normalise to list of paths
        if isinstance(result, str):
            return [result]
        if isinstance(result, (list, tuple, set)):
            return [str(p) for p in result]

        # Some implementations may return generators / other iterables
        try:
            return [str(p) for p in result]  # type: ignore[arg-type]
        except TypeError:
            return []

    raise RuntimeError(
        "Incompatible eumdac API: product object has no 'download' method. "
        "Install a compatible version, e.g. 'eumdac<3.0.0', or update "
        "contrai.data.mtg_fci to match your eumdac version."
    )


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

    Raises
    ------
    ValueError
        If search parameters are invalid.
    RuntimeError
        If the installed ``eumdac`` version is incompatible with the
        expected download API.
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
