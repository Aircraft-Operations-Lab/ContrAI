
import numpy as np
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plotly.graph_objects as go


def plot_overlay_interactive_light(overlay, title: str = "Contrail Overlay", max_size: int = 1200):
    """
    Lightweight interactive zoom/pan viewer for an overlay image using Plotly.

    Parameters
    ----------
    overlay : np.ndarray or PIL.Image.Image or xarray-like
        RGB image, shape (H, W, 3) or convertible to that.
    title : str
        Figure title.
    max_size : int
        Maximum size (in pixels) of the longest image dimension after downsampling.
    """
    # --- 1. Convert to NumPy array ---
    if isinstance(overlay, Image.Image):
        img = np.array(overlay)
    else:
        img = np.array(overlay)  # handles np.ndarray, xarray.DataArray, etc.

    # Ensure it's 3-channel
    if img.ndim == 2:  # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]

    h, w = img.shape[:2]

    # --- 2. Downsample if too large (to keep Plotly light) ---
    max_dim = max(h, w)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
        img = np.array(img_pil)
        h, w = img.shape[:2]

    # --- 3. Build Plotly figure ---
    fig = go.Figure(data=[go.Image(z=img)])

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=22, color="white"),
        ),
        width=min(900, w + 100),
        height=min(900, h + 100),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        dragmode="pan",
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")

    fig.show(config={"scrollZoom": True})
import numpy as np
import plotly.graph_objects as go

def plot_contrails_plotly_geo_orange(
    geojson_fc,
    title="Contrail Detection",
    subtitle="Interactive polygon view",
    max_points_per_polygon: int = 300,
):
    fig = go.Figure()
    all_lons = []
    all_lats = []

    for feature in geojson_fc["features"]:
        geom = feature.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue

        rings = geom.get("coordinates", [])
        if not rings:
            continue

        shell = rings[0]
        if len(shell) < 4:
            continue

        if len(shell) > max_points_per_polygon:
            idx = np.linspace(0, len(shell) - 1, max_points_per_polygon, dtype=int)
            shell = [shell[i] for i in idx]

        lons, lats = zip(*shell)
        all_lons.extend(lons)
        all_lats.extend(lats)

        feature_id = feature.get("properties", {}).get("id", "contrail")

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",          # 👈 only lines
                # fill=None or "none"   # 👈 no fill
                line=dict(width=1),
                name=str(feature_id),
                hoverinfo="text",
                text=str(feature_id),
            )
        )

    if not all_lons or not all_lats:
        raise ValueError("No valid polygon coordinates to plot.")

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    pad_lon = max(1.0, 0.05 * (max_lon - min_lon))
    pad_lat = max(1.0, 0.05 * (max_lat - min_lat))

    fig.update_geos(
        projection_type="natural earth",  # or "equirectangular" / "mercator"
        lonaxis_range=[min_lon - pad_lon, max_lon + pad_lon],
        lataxis_range=[min_lat - pad_lat, max_lat + pad_lat],
        showcoastlines=True,
        showcountries=True,
    )

    fig.update_layout(
        paper_bgcolor="rgba(5, 5, 15, 1.0)",
        plot_bgcolor="rgba(5, 5, 15, 1.0)",
        title=dict(
            text=(
                f"<b>{title}</b>"
                f"<br><span style='font-size:13px; color:#AAAAAA;'>{subtitle}</span>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=720,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    return fig


def plot_contrails_plotly_geo_dark(

    geojson_fc,
    title="Contrail Detection",
    subtitle="Interactive polygon view",
    max_points_per_polygon: int = 300,
):
    fig = go.Figure()
    all_lons = []
    all_lats = []

    # Pastel colors — light enough to show on dark background
    pastel_colors = [
        "rgba(255, 182, 193, 0.95)",  # pink
        "rgba(174, 198, 207, 0.95)",  # light steel blue
        "rgba(255, 223, 186, 0.95)",  # soft peach
        "rgba(210, 225, 168, 0.95)",  # spring green
        "rgba(222, 203, 250, 0.95)",  # lavender
        "rgba(194, 255, 232, 0.95)",  # mint aqua
    ]
    i = 0

    for feature in geojson_fc["features"]:
        geom = feature.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue

        rings = geom.get("coordinates", [])
        if not rings:
            continue

        shell = rings[0]
        if len(shell) < 4:
            continue

        if len(shell) > max_points_per_polygon:
            idx = np.linspace(0, len(shell) - 1, max_points_per_polygon, dtype=int)
            shell = [shell[i] for i in idx]

        lons, lats = zip(*shell)
        all_lons.extend(lons)
        all_lats.extend(lats)

        # Select pastel color (cycled)
        color = pastel_colors[i % len(pastel_colors)]
        i += 1

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=1.2, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if not all_lons or not all_lats:
        raise ValueError("No valid polygon coordinates to plot.")

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    pad_lon = max(1.0, 0.05 * (max_lon - min_lon))
    pad_lat = max(1.0, 0.05 * (max_lat - min_lat))

    fig.update_geos(
        projection_type="natural earth",
        lonaxis_range=[min_lon - pad_lon, max_lon + pad_lon],
        lataxis_range=[min_lat - pad_lat, max_lat + pad_lat],
        showcoastlines=True,
        coastlinecolor="rgba(200, 200, 200, 0.7)",
        showcountries=True,
        countrycolor="rgba(160, 160, 160, 0.7)",
        showland=True,
        landcolor="rgba(30, 30, 40, 1.0)",
        showocean=True,
        oceancolor="rgba(5, 5, 20, 1.0)",
        lonaxis_showgrid=True,
        lataxis_showgrid=True,
        lonaxis_gridcolor="rgba(120, 120, 150, 0.3)",
        lataxis_gridcolor="rgba(120, 120, 150, 0.3)",
    )

    fig.update_layout(
        paper_bgcolor="rgba(5, 5, 15, 1.0)",
        plot_bgcolor="rgba(5, 5, 15, 1.0)",
        title=dict(
            text=(
                f"<b>{title}</b>"
                f"<br><span style='font-size:13px; color:#AAAAAA;'>{subtitle}</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(color="#FFFFFF", size=22),
        ),
        font=dict(color="#EEEEEE"),
        height=720,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    return fig


def plot_contrails_plotly_geo_pastel(
    geojson_fc,
    title="Contrail Detection",
    subtitle="Interactive polygon view",
    max_points_per_polygon: int = 300,
):
    fig = go.Figure()
    all_lons = []
    all_lats = []

    pastel_colors = [
        "rgba(255,182,193,0.95)",  # pink
        "rgba(174,198,207,0.95)",  # light steel blue
        "rgba(255,223,186,0.95)",  # peach
        "rgba(210,225,168,0.95)",  # spring green
        "rgba(222,203,250,0.95)",  # lavender
        "rgba(194,255,232,0.95)",  # mint aqua
    ]
    i = 0

    for feature in geojson_fc["features"]:
        geom = feature.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue

        rings = geom.get("coordinates", [])
        if not rings:
            continue

        shell = rings[0]
        if len(shell) < 4:
            continue

        if len(shell) > max_points_per_polygon:
            idx = np.linspace(0, len(shell) - 1, max_points_per_polygon, dtype=int)
            shell = [shell[j] for j in idx]

        lons, lats = zip(*shell)
        all_lons.extend(lons)
        all_lats.extend(lats)

        feature_id = feature.get("properties", {}).get("id", f"C{i}")
        centroid_lon = float(np.mean(lons))
        centroid_lat = float(np.mean(lats))

        hover_text = (
            f"ID: {feature_id}"
            f"<br>lon: {centroid_lon:.3f}"
            f"<br>lat: {centroid_lat:.3f}"
        )

        color = pastel_colors[i % len(pastel_colors)]
        i += 1

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(color=color, width=1.3),
                name=str(feature_id),   # legend label
                text=hover_text,        # what we want to see on hover
                hoverinfo="text",       # 👈 use text only, no template
                showlegend=True,
            )
        )

    if not all_lons or not all_lats:
        raise ValueError("No valid polygon coordinates to plot.")

    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    pad_lon = max(1.0, 0.05 * (max_lon - min_lon))
    pad_lat = max(1.0, 0.05 * (max_lat - min_lat))

    fig.update_geos(
        projection_type="natural earth",
        lonaxis_range=[min_lon - pad_lon, max_lon + pad_lon],
        lataxis_range=[min_lat - pad_lat, max_lat + pad_lat],
        showcoastlines=True,
        coastlinecolor="rgba(200,200,200,0.7)",
        showcountries=True,
        countrycolor="rgba(160,160,160,0.7)",
        showland=True,
        landcolor="rgba(30,30,40,1.0)",
        showocean=True,
        oceancolor="rgba(5,5,20,1.0)",
        lonaxis_showgrid=True,
        lataxis_showgrid=True,
        lonaxis_gridcolor="rgba(120,120,150,0.3)",
        lataxis_gridcolor="rgba(120,120,150,0.3)",
    )

    fig.update_layout(
        paper_bgcolor="rgba(5,5,15,1.0)",
        plot_bgcolor="rgba(5,5,15,1.0)",
        title=dict(
            text=(
                f"<b>{title}</b>"
                f"<br><span style='font-size:13px; color:#AAAAAA;'>{subtitle}</span>"
            ),
            x=0.5,
            font=dict(color="#FFFFFF", size=22),
        ),
        font=dict(color="#EEEEEE"),
        legend=dict(
            bgcolor="rgba(10,10,30,0.85)",
            bordercolor="rgba(200,200,200,0.25)",
            borderwidth=1,
            font=dict(size=10),
        ),
        height=720,
        margin=dict(l=10, r=10, t=80, b=10),
    )

    return fig
