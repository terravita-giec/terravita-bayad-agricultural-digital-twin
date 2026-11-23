# ============================================================
# TERRAVITA | MODULE 06
# BAYAD AGRICULTURAL DIGITAL TWIN DASHBOARD
# Google Colab + Streamlit
# ============================================================

import os
import io
import json
import zipfile
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import rasterio
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 01. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TerraVita | Bayad Agricultural Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 02. PROFESSIONAL NAVY / TEAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #081522;
        color: #E8F1F5;
    }
    [data-testid="stSidebar"] {
        background: #0D2030;
        border-right: 1px solid #18394A;
    }
    [data-testid="stHeader"] {
        background: rgba(8,21,34,0.95);
    }
    h1, h2, h3 {
        color: #F1F7FA !important;
    }
    p, label, .stMarkdown {
        color: #D3E0E7;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #102B3B, #0D2231);
        border: 1px solid #1C4655;
        border-radius: 13px;
        padding: 15px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #9CB8C5;
    }
    div[data-testid="stMetricValue"] {
        color: #42D6C5;
    }
    .tv-hero {
        background: linear-gradient(115deg, #102B3B 0%, #0C2635 55%, #103D43 100%);
        border: 1px solid #1D5660;
        padding: 26px 30px;
        border-radius: 18px;
        margin-bottom: 18px;
    }
    .tv-kicker {
        color: #49D8C7;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 800;
        text-transform: uppercase;
    }
    .tv-subtitle {
        color: #B6CCD5;
        font-size: 14px;
        line-height: 1.6;
    }
    .tv-section {
        color: #49D8C7;
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 12px;
        text-transform: uppercase;
    }
    .tv-note {
        background: #102A39;
        border-left: 4px solid #35BDB0;
        padding: 12px 16px;
        border-radius: 8px;
        color: #D5E7EC;
        font-size: 13px;
    }
    .stDownloadButton button {
        background: #0D8F87;
        color: white;
        border: 0;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton button:hover {
        background: #12A99D;
        color: white;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #1C4655;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 03. PROJECT PATHS
# ============================================================

DEFAULT_PROJECT = "/content/drive/MyDrive/TERRAVITA_BAYAD"

st.sidebar.markdown(
    """
    <div style="padding:8px 0 18px 0;">
      <div style="font-size:12px;letter-spacing:2px;color:#49D8C7;font-weight:800;">
        TERRAVITA
      </div>
      <div style="font-size:21px;font-weight:800;color:#F1F7FA;">
        AGRI DIGITAL TWIN
      </div>
      <div style="font-size:11px;color:#9CB8C5;">
        BAYAD TALUKA · GUJARAT
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

PROJECT = st.sidebar.text_input(
    "Google Drive project folder",
    value=DEFAULT_PROJECT
).strip()

if not PROJECT:
    PROJECT = DEFAULT_PROJECT

OUTPUT_02 = os.path.join(PROJECT, "TERRAVITA_MODULE_02")
OUTPUT_03_TERRAIN = os.path.join(PROJECT, "TERRAVITA_MODULE_03_TERRAIN")
OUTPUT_03_RAIN = os.path.join(PROJECT, "TERRAVITA_MODULE_03_RAINFALL")
OUTPUT_04 = os.path.join(PROJECT, "TERRAVITA_MODULE_04", "01_OUTPUTS")
OUTPUT_05 = os.path.join(PROJECT, "TERRAVITA_MODULE_05", "01_OUTPUTS")

# The boundary path in Module 01 may vary depending on where its
# outputs were exported. The app checks these known alternatives.
PATHS = {
    "boundary": [
        os.path.join(PROJECT, "01_OUTPUTS", "Bayad_Taluka_2025.geojson"),
        os.path.join(PROJECT, "01_OUTPUTS", "Bayad_Taluka_2025.gpkg"),
        os.path.join(PROJECT, "TERRAVITA_MODULE_01", "01_OUTPUTS", "Bayad_Taluka_2025.geojson"),
        os.path.join(PROJECT, "TERRAVITA_MODULE_01", "01_OUTPUTS", "Bayad_Taluka_2025.gpkg"),
        os.path.join(PROJECT, "Bayad_Taluka_2025.geojson"),
    ],
    "indices": [
        os.path.join(OUTPUT_02, "Bayad_Kharif_2025_Indices.tif"),
    ],
    "terrain": [
        os.path.join(OUTPUT_03_TERRAIN, "Bayad_SRTM_Terrain_30m.tif"),
    ],
    "rainfall": [
        os.path.join(OUTPUT_03_RAIN, "Bayad_Kharif_2025_Rainfall.tif"),
        os.path.join(PROJECT, "TERRAVITA_MODULE_03", "Bayad_Kharif_2025_Rainfall.tif"),
    ],
    "stress": [
        os.path.join(OUTPUT_04, "Bayad_Kharif2025_Relative_Stress.tif"),
    ],
    "stress_classes": [
        os.path.join(OUTPUT_04, "Bayad_Kharif2025_Stress_Classes.tif"),
    ],
    "clusters": [
        os.path.join(OUTPUT_04, "Bayad_Kharif2025_GeoAI_Clusters.tif"),
    ],
    "drainage_distance": [
        os.path.join(OUTPUT_04, "Bayad_Kharif2025_Drainage_Distance.tif"),
    ],
    "priority": [
        os.path.join(OUTPUT_05, "Bayad_Intervention_Priority.tif"),
    ],
    "priority_classes": [
        os.path.join(OUTPUT_05, "Bayad_Intervention_Priority_Classes.tif"),
    ],
    "candidates_gpkg": [
        os.path.join(OUTPUT_05, "Bayad_Intervention_Candidates.gpkg"),
    ],
    "candidates_csv": [
        os.path.join(OUTPUT_05, "Bayad_Intervention_Candidates.csv"),
    ],
    "scenario_csv": [
        os.path.join(OUTPUT_05, "Bayad_Intervention_Scenario_Results.csv"),
    ],
    "statistics_json": [
        os.path.join(OUTPUT_05, "Bayad_Module05_Statistics.json"),
    ],
    "scenario_buffers": [
        os.path.join(OUTPUT_05, "Bayad_Intervention_Scenario_Buffers.gpkg"),
    ],
}

def resolve_path(key):
    for candidate in PATHS.get(key, []):
        if os.path.exists(candidate):
            return candidate
    return None

RESOLVED = {key: resolve_path(key) for key in PATHS}

# ============================================================
# 04. CACHED DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_boundary(path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Boundary file contains no features.")
    if gdf.crs is None:
        raise ValueError("Boundary file has no CRS defined.")
    if not gdf.geometry.notna().any():
        raise ValueError("Boundary file has no valid geometries.")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.is_valid | gdf.geometry.notna()].copy()
    return gdf

@st.cache_data(show_spinner=False)
def read_raster_preview(path, band=1, max_size=1000):
    with rasterio.open(path) as src:
        if src.count < band:
            raise ValueError(
                f"Raster has {src.count} band(s), but band {band} was requested."
            )
        if src.crs is None:
            raise ValueError("Raster CRS is missing.")
        if src.width < 1 or src.height < 1:
            raise ValueError("Raster has invalid dimensions.")

        scale = min(1.0, max_size / max(src.width, src.height))
        out_width = max(1, int(src.width * scale))
        out_height = max(1, int(src.height * scale))

        arr = src.read(
            band,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=Resampling.nearest if band in [5] else Resampling.bilinear
        ).astype("float32")

        arr = np.asarray(arr.filled(np.nan), dtype="float32")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21
        )

        meta = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs),
            "bounds": tuple(float(x) for x in bounds),
            "nodata": src.nodata,
            "dtype": src.dtypes[band - 1],
            "transform": tuple(src.transform),
            "descriptions": list(src.descriptions),
        }

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Raster band contains no valid pixels.")

    return arr, meta

@st.cache_data(show_spinner=False)
def load_table(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_vector(path, layer=None):
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Vector file contains no features.")
    if gdf.crs is None:
        raise ValueError("Vector file CRS is missing.")
    return gdf

# ============================================================
# 05. DATA AVAILABILITY AND VALIDATION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### Data availability")

for key, path in RESOLVED.items():
    if path:
        st.sidebar.success(f"{key.replace('_', ' ').title()}: ready")
    else:
        st.sidebar.warning(f"{key.replace('_', ' ').title()}: not found")

def safe_boundary():
    path = RESOLVED["boundary"]
    if not path:
        return None, "Boundary file not found."
    try:
        return load_boundary(path), None
    except Exception as e:
        return None, str(e)

def safe_raster(key, band=1):
    path = RESOLVED.get(key)
    if not path:
        return None, None, f"{key} raster not found."
    try:
        arr, meta = read_raster_preview(path, band=band)
        return arr, meta, None
    except Exception as e:
        return None, None, str(e)

def safe_vector(key):
    path = RESOLVED.get(key)
    if not path:
        return None, f"{key} vector file not found."
    try:
        return load_vector(path), None
    except Exception as e:
        return None, str(e)

def safe_csv(key):
    path = RESOLVED.get(key)
    if not path:
        return None, f"{key} CSV not found."
    try:
        return load_table(path), None
    except Exception as e:
        return None, str(e)

def safe_json(key):
    path = RESOLVED.get(key)
    if not path:
        return None, f"{key} JSON not found."
    try:
        return load_json(path), None
    except Exception as e:
        return None, str(e)

boundary, boundary_error = safe_boundary()

# ============================================================
# 06. GENERAL HELPERS
# ============================================================

def finite_stats(arr):
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return None
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "valid_pixels": int(values.size)
    }

def format_number(value, digits=2):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.{digits}f}"

def raster_image_data_uri(arr, cmap="viridis", vmin=None, vmax=None):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    if vmin is None:
        vmin = float(np.nanpercentile(finite, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def add_raster_overlay(map_object, arr, meta, name, cmap="viridis", opacity=0.65):
    west, south, east, north = meta["bounds"]
    image_uri = raster_image_data_uri(arr, cmap=cmap)
    if image_uri:
        folium.raster_layers.ImageOverlay(
            image=image_uri,
            bounds=[[south, west], [north, east]],
            opacity=opacity,
            name=name,
            interactive=True,
            cross_origin=False,
            zindex=2
        ).add_to(map_object)

def make_base_map(boundary_gdf=None, zoom=10):
    if boundary_gdf is not None and not boundary_gdf.empty:
        b = boundary_gdf.to_crs("EPSG:4326")
        centroid = b.geometry.union_all().centroid
        center = [centroid.y, centroid.x]
    else:
        center = [23.42, 73.28]

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True
    )

    if boundary_gdf is not None and not boundary_gdf.empty:
        b = boundary_gdf.to_crs("EPSG:4326")
        folium.GeoJson(
            b,
            name="Bayad Taluka boundary",
            style_function=lambda feature: {
                "color": "#28C7B7",
                "weight": 3,
                "fillColor": "#28C7B7",
                "fillOpacity": 0.05
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    c for c in ["NAME", "District", "Sub_dist"]
                    if c in b.columns
                ]
            ) if any(c in b.columns for c in ["NAME", "District", "Sub_dist"]) else None
        ).add_to(m)

    return m

def show_map(m, height=600):
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(
        m,
        width=None,
        height=height,
        returned_objects=[]
    )

def make_pdf_report(summary, scenario_df=None):
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        plt.axis("off")

        lines = [
            "TERRAVITA ENVIRONMENTAL & GEOSPATIAL CONSULTANCY",
            "",
            "BAYAD AGRICULTURAL DIGITAL TWIN",
            "Module 06 | Interactive Decision-Support Dashboard",
            "",
            "Study area: Bayad Taluka, Aravalli, Gujarat, India",
            "",
            "PROJECT SUMMARY",
            "This report summarizes available outputs from Modules 01–05.",
            "",
        ]

        for key, value in summary.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")

        lines += [
            "",
            "INTERPRETATION AND LIMITATIONS",
            "Priority scores are screening indicators based on the configured",
            "weighted overlay. Scenario coverage measures spatial overlap of",
            "candidate buffers with high-priority pixels. It does not estimate",
            "groundwater recharge, water storage, crop-yield gains, or measured",
            "intervention effectiveness.",
            "",
            "Candidate locations require field, land-access, soil, geology,",
            "hydrogeology and engineering validation before implementation.",
            "",
            "Prepared using the TerraVita Bayad project data available",
            "at the time this dashboard was run."
        ]

        y = 0.95
        for i, line in enumerate(lines):
            fontsize = 15 if i in [0, 2, 7] else 10
            weight = "bold" if i in [0, 2, 7] else "normal"
            fig.text(
                0.09, y, line,
                fontsize=fontsize,
                fontweight=weight,
                color="#102B3B"
            )
            y -= 0.035 if line else 0.02

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if scenario_df is not None and not scenario_df.empty:
            fig, ax = plt.subplots(figsize=(8.27, 5.5))
            ax.bar(
                scenario_df["scenario_radius_m"].astype(str),
                scenario_df["coverage_percent"]
            )
            ax.set_title("Intervention Scenario Coverage")
            ax.set_xlabel("Buffer radius (m)")
            ax.set_ylabel("High-priority pixel coverage (%)")
            ax.grid(axis="y", alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ============================================================
# 07. HEADER
# ============================================================

st.markdown(
    """
    <div class="tv-hero">
      <div class="tv-kicker">TERRAVITA · GEOSPATIAL INTELLIGENCE</div>
      <h1 style="margin:8px 0 6px 0;font-size:30px;">
        Agricultural Digital Twin
      </h1>
      <div style="font-size:16px;color:#D8E9EE;font-weight:600;">
        Watershed Resilience & Agricultural Stress Intelligence
      </div>
      <div class="tv-subtitle" style="margin-top:10px;">
        Bayad Taluka · Aravalli District · Gujarat, India
        <br>
        Satellite Remote Sensing · GIS · GeoAI · Intervention Scenario Screening
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 08. NAVIGATION
# ============================================================

PAGES = [
    "Overview",
    "GIS Map",
    "Satellite Indices",
    "Agricultural Stress",
    "GeoAI Clusters",
    "Digital Twin Scenarios",
    "Reports & Downloads",
    "Data Quality"
]

page = st.sidebar.radio("DASHBOARD NAVIGATION", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption("TerraVita | Satellite + GIS + AI")
st.sidebar.caption("Analytical prototype · Not a live sensor system")

# ============================================================
# 09. OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown('<div class="tv-section">01 / Project overview</div>', unsafe_allow_html=True)
    st.header("Bayad Taluka Monitoring Overview")

    if boundary is None:
        st.warning(f"Boundary unavailable: {boundary_error}")

    stress_arr, stress_meta, stress_err = safe_raster("stress")
    priority_arr, priority_meta, priority_err = safe_raster("priority")
    candidate_gdf, candidate_err = safe_vector("candidates_gpkg")
    stats_json, stats_err = safe_json("statistics_json")

    stress_stats = finite_stats(stress_arr) if stress_arr is not None else None
    priority_stats = finite_stats(priority_arr) if priority_arr is not None else None

    area_km2 = None
    if boundary is not None:
        try:
            area_km2 = float(
                boundary.to_crs("EPSG:32643").geometry.area.sum() / 1e6
            )
        except Exception:
            area_km2 = None

    candidate_count = (
        len(candidate_gdf)
        if candidate_gdf is not None else
        (stats_json.get("candidate_count") if stats_json else None)
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Study area", f"{area_km2:,.1f} km²" if area_km2 else "N/A")
    k2.metric(
        "Mean relative stress",
        format_number(stress_stats["mean"]) if stress_stats else "N/A"
    )
    k3.metric(
        "Mean priority score",
        format_number(priority_stats["mean"]) if priority_stats else "N/A"
    )
    k4.metric(
        "Candidate locations",
        f"{candidate_count:,}" if candidate_count is not None else "N/A"
    )

    st.markdown("### Project workflow")
    cols = st.columns(5)
    modules = [
        ("01", "AOI verification", "Boundary"),
        ("02", "Satellite indices", "NDVI · NDWI · NDMI"),
        ("03", "Rainfall & terrain", "CHIRPS · SRTM"),
        ("04", "Stress & GeoAI", "Stress · KMeans"),
        ("05", "Digital Twin", "Priority · Scenarios")
    ]
    for col, item in zip(cols, modules):
        with col:
            st.markdown(
                f"""
                <div style="background:#102A39;border:1px solid #1C4655;
                border-radius:12px;padding:14px;min-height:120px;">
                  <div style="color:#49D8C7;font-size:20px;font-weight:800;">{item[0]}</div>
                  <div style="font-weight:700;color:#F1F7FA;margin-top:6px;">{item[1]}</div>
                  <div style="font-size:11px;color:#9CB8C5;margin-top:6px;">{item[2]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Priority screening map")
    if priority_arr is not None and priority_meta is not None:
        m = make_base_map(boundary)
        add_raster_overlay(
            m, priority_arr, priority_meta,
            "Intervention priority score", cmap="YlOrRd", opacity=0.72
        )
        if candidate_gdf is not None:
            cg = candidate_gdf.to_crs("EPSG:4326")
            for _, row in cg.iterrows():
                folium.CircleMarker(
                    [row.geometry.y, row.geometry.x],
                    radius=5,
                    color="#0B6B63",
                    fill=True,
                    fill_opacity=0.95,
                    tooltip=str(row.get("candidate_id", "Candidate"))
                ).add_to(m)
        show_map(m, 560)
    else:
        st.info(f"Priority raster unavailable: {priority_err}")

    st.markdown(
        """
        <div class="tv-note">
        This dashboard presents a spatial screening prototype using the available
        Module 01–05 outputs. It is not a real-time sensor-connected or physically
        calibrated watershed simulation.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 10. GIS MAP
# ============================================================

elif page == "GIS Map":

    st.markdown('<div class="tv-section">02 / Spatial intelligence</div>', unsafe_allow_html=True)
    st.header("Interactive GIS Map")

    if boundary is None:
        st.error(f"Boundary cannot be loaded: {boundary_error}")
    else:
        raster_options = {
            "Intervention priority": ("priority", 1, "YlOrRd"),
            "Relative agricultural stress": ("stress", 1, "YlOrRd"),
            "Stress classes": ("stress_classes", 1, "Set1"),
            "GeoAI clusters": ("clusters", 1, "tab10"),
            "Terrain elevation": ("terrain", 1, "terrain"),
            "Terrain slope": ("terrain", 2, "magma"),
            "Drainage distance": ("drainage_distance", 1, "Blues"),
        }

        available = []
        for label, (key, band, cmap) in raster_options.items():
            if RESOLVED.get(key):
                available.append(label)

        selected_layer = st.selectbox(
            "Select map layer",
            ["Boundary only"] + available
        )

        m = make_base_map(boundary, zoom=10)

        if selected_layer != "Boundary only":
            key, band, cmap = raster_options[selected_layer]
            arr, meta, err = safe_raster(key, band)
            if arr is not None:
                add_raster_overlay(
                    m, arr, meta, selected_layer, cmap=cmap, opacity=0.68
                )
            else:
                st.warning(err)

        candidates, candidates_error = safe_vector("candidates_gpkg")
        if candidates is not None:
            candidates = candidates.to_crs("EPSG:4326")
            for _, row in candidates.iterrows():
                popup = (
                    f"Candidate: {row.get('candidate_id', 'N/A')}<br>"
                    f"Rank: {row.get('priority_rank', 'N/A')}<br>"
                    f"Score: {row.get('priority_score', 'N/A')}"
                )
                folium.Marker(
                    [row.geometry.y, row.geometry.x],
                    tooltip=str(row.get("candidate_id", "Candidate")),
                    popup=folium.Popup(popup, max_width=280),
                    icon=folium.Icon(color="green", icon="info-sign")
                ).add_to(m)

        buffers, buffer_error = safe_vector("scenario_buffers")
        if buffers is not None:
            buffers = buffers.to_crs("EPSG:4326")
            radius_values = sorted(
                [v for v in buffers["radius_m"].dropna().unique()]
            ) if "radius_m" in buffers.columns else []
            selected_radius = st.selectbox(
                "Scenario buffer radius",
                ["All"] + [str(int(v)) for v in radius_values],
                key="map_radius"
            )
            if selected_radius != "All" and "radius_m" in buffers.columns:
                buffers = buffers[
                    buffers["radius_m"] == int(selected_radius)
                ]
            folium.GeoJson(
                buffers,
                name="Intervention buffers",
                style_function=lambda feature: {
                    "color": "#E5A84B",
                    "weight": 1.5,
                    "fillColor": "#E5A84B",
                    "fillOpacity": 0.10
                }
            ).add_to(m)

        show_map(m, 650)

# ============================================================
# 11. SATELLITE INDICES
# ============================================================

elif page == "Satellite Indices":

    st.markdown('<div class="tv-section">03 / Earth observation</div>', unsafe_allow_html=True)
    st.header("Satellite Indicator Explorer")

    indices_path = RESOLVED["indices"]

    if not indices_path:
        st.error("Sentinel-2 index raster not found.")
        st.code(os.path.join(OUTPUT_02, "Bayad_Kharif_2025_Indices.tif"))
    else:
        with rasterio.open(indices_path) as src:
            band_count = src.count
            descriptions = list(src.descriptions)

        band_names = {
            1: "NDVI",
            2: "NDWI",
            3: "MNDWI",
            4: "NDMI",
            5: "Valid observations"
        }

        options = {
            f"Band {i} – {band_names.get(i, 'Unknown')}"
            + (f" ({descriptions[i-1]})" if descriptions[i-1] else ""): i
            for i in range(1, band_count + 1)
        }

        selected = st.selectbox(
            "Select satellite indicator",
            list(options.keys())
        )
        band = options[selected]

        arr, meta, err = safe_raster("indices", band)

        if arr is None:
            st.error(err)
        else:
            stats = finite_stats(arr)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Minimum", format_number(stats["min"], 3))
            c2.metric("Maximum", format_number(stats["max"], 3))
            c3.metric("Mean", format_number(stats["mean"], 3))
            c4.metric("Valid pixels", f"{stats['valid_pixels']:,}")

            cmap = "viridis"
            if band == 1:
                cmap = "RdYlGn"
            elif band in [2, 3, 4]:
                cmap = "BrBG"
            elif band == 5:
                cmap = "Blues"

            fig, ax = plt.subplots(figsize=(10, 6))
            image = ax.imshow(arr, cmap=cmap)
            ax.set_title(selected)
            ax.axis("off")
            fig.colorbar(image, ax=ax, shrink=0.75)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                f"Source: {os.path.basename(indices_path)} | "
                f"CRS: {meta['crs']} | "
                f"Original dimensions: {meta['width']} × {meta['height']}"
            )

            st.markdown(
                """
                <div class="tv-note">
                NDVI represents vegetation greenness; NDWI/MNDWI are spectral
                water indicators; NDMI is a vegetation moisture-related index.
                These indices are not direct measurements of crop yield,
                groundwater level or field soil moisture.
                </div>
                """,
                unsafe_allow_html=True
            )
# ============================================================
# 12. AGRICULTURAL STRESS
# ============================================================

elif page == "Agricultural Stress":

    st.markdown('<div class="tv-section">04 / Agricultural condition</div>', unsafe_allow_html=True)
    st.header("Agricultural Stress Screening")

    stress_arr, stress_meta, stress_err = safe_raster("stress")
    classes_arr, classes_meta, classes_err = safe_raster("stress_classes")

    if stress_arr is None:
        st.error(stress_err)
    else:
        stats = finite_stats(stress_arr)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean relative stress", format_number(stats["mean"], 3))
        c2.metric("Median relative stress", format_number(stats["median"], 3))
        c3.metric("Valid pixels", f"{stats['valid_pixels']:,}")

        tab1, tab2 = st.tabs(["Relative stress", "Stress classes"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(stress_arr, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_title("Kharif 2025 – Relative Agricultural Stress")
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="Relative stress score")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            if classes_arr is None:
                st.info(classes_err)
            else:
                vals, counts = np.unique(
                    classes_arr[np.isfinite(classes_arr)],
                    return_counts=True
                )
                class_df = pd.DataFrame({
                    "Class code": vals.astype(int),
                    "Pixel count": counts
                })
                class_labels = {
                    1: "Lower relative stress",
                    2: "Moderate relative stress",
                    3: "Elevated relative stress",
                    4: "Higher relative stress"
                }
                class_df["Interpretation"] = class_df["Class code"].map(
                    class_labels
                ).fillna("Other / background")
                st.dataframe(class_df, use_container_width=True)

                fig = px.bar(
                    class_df,
                    x="Interpretation",
                    y="Pixel count",
                    title="Agricultural stress class distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="tv-note">
            These are relative, within-season screening values derived from the
            project workflow. They are not calibrated drought categories or
            independently validated crop-health diagnoses.
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 13. GEOAI CLUSTERS
# ============================================================

elif page == "GeoAI Clusters":

    st.markdown('<div class="tv-section">05 / Unsupervised spatial analytics</div>', unsafe_allow_html=True)
    st.header("GeoAI Agricultural Clusters")

    clusters, cluster_meta, cluster_err = safe_raster("clusters")

    if clusters is None:
        st.error(cluster_err)
    else:
        cluster_values = clusters[np.isfinite(clusters)]
        unique_values, counts = np.unique(
            cluster_values.astype(int),
            return_counts=True
        )

        cluster_df = pd.DataFrame({
            "Cluster": unique_values,
            "Pixel count": counts
        })
        cluster_df["Share (%)"] = (
            cluster_df["Pixel count"] /
            cluster_df["Pixel count"].sum() * 100
        )

        c1, c2 = st.columns([1.4, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(9, 6))
            im = ax.imshow(clusters, cmap="tab10")
            ax.set_title("Kharif 2025 – GeoAI Cluster Map")
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="Cluster ID")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("### Cluster distribution")
            st.dataframe(
                cluster_df.round(2),
                use_container_width=True,
                hide_index=True
            )
            fig = px.pie(
                cluster_df,
                names="Cluster",
                values="Pixel count",
                hole=0.45
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="tv-note">
            KMeans clusters are exploratory groupings based on the input features.
            Cluster IDs are labels, not ordered severity classes. They should not
            be interpreted as validated crop types or ground-truth agricultural
            conditions without field observations.
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 14. DIGITAL TWIN SCENARIOS
# ============================================================

elif page == "Digital Twin Scenarios":

    st.markdown('<div class="tv-section">06 / Intervention planning</div>', unsafe_allow_html=True)
    st.header("Digital Twin – Intervention Scenario Simulator")

    priority, priority_meta, priority_err = safe_raster("priority")
    candidates, candidates_err = safe_vector("candidates_gpkg")
    scenario_df, scenario_err = safe_csv("scenario_csv")
    stats_json, stats_err = safe_json("statistics_json")

    if priority is None:
        st.error(priority_err)
    else:
        stats = finite_stats(priority)
        k1, k2, k3 = st.columns(3)
        k1.metric("Mean priority score", format_number(stats["mean"], 3))
        k2.metric("Maximum priority score", format_number(stats["max"], 3))
        k3.metric("Valid priority pixels", f"{stats['valid_pixels']:,}")

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(priority, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title("Intervention Priority Index")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Priority score")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Candidate intervention locations")
    if candidates is not None:
        st.dataframe(
            candidates.drop(columns="geometry", errors="ignore"),
            use_container_width=True,
            hide_index=True
        )
        candidates_wgs = candidates.to_crs("EPSG:4326")
        csv_df = candidates_wgs.copy()
        csv_df["longitude"] = csv_df.geometry.x
        csv_df["latitude"] = csv_df.geometry.y
        st.download_button(
            "Download candidate locations (CSV)",
            csv_df.drop(columns="geometry").to_csv(index=False).encode("utf-8"),
            file_name="TerraVita_Bayad_Intervention_Candidates.csv",
            mime="text/csv"
        )
    else:
        st.info(candidates_err)

    st.markdown("### Scenario coverage comparison")
    if scenario_df is not None:
        required = {
            "scenario_radius_m",
            "coverage_percent"
        }
        if required.issubset(scenario_df.columns):
            fig = px.bar(
                scenario_df,
                x="scenario_radius_m",
                y="coverage_percent",
                text="coverage_percent",
                title="High-priority pixel coverage by scenario"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                xaxis_title="Buffer radius (m)",
                yaxis_title="High-priority pixel coverage (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(scenario_df, use_container_width=True)
        else:
            st.warning(
                "Scenario CSV does not contain the expected columns: "
                "scenario_radius_m and coverage_percent."
            )
    else:
        st.info(scenario_err)

    st.markdown(
        """
        <div class="tv-note">
        Scenario coverage is a geometric overlap calculation between candidate
        buffers and high-priority pixels. It does not simulate hydrological
        connectivity, recharge, storage, crop response or intervention outcomes.
        The candidate points are investigation priorities, not construction
        approvals.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 15. REPORTS AND DOWNLOADS
# ============================================================

elif page == "Reports & Downloads":

    st.markdown('<div class="tv-section">07 / Reporting</div>', unsafe_allow_html=True)
    st.header("Reports & Data Downloads")

    download_items = [
        ("Boundary GeoJSON", "boundary"),
        ("Satellite indices GeoTIFF", "indices"),
        ("Terrain GeoTIFF", "terrain"),
        ("Agricultural stress GeoTIFF", "stress"),
        ("Stress classes GeoTIFF", "stress_classes"),
        ("GeoAI clusters GeoTIFF", "clusters"),
        ("Intervention priority GeoTIFF", "priority"),
        ("Priority classes GeoTIFF", "priority_classes"),
        ("Candidate locations GeoPackage", "candidates_gpkg"),
        ("Candidate locations CSV", "candidates_csv"),
        ("Scenario results CSV", "scenario_csv"),
        ("Scenario buffers GeoPackage", "scenario_buffers"),
        ("Module 05 statistics JSON", "statistics_json"),
    ]

    st.markdown("### Available project files")

    for label, key in download_items:
        path = RESOLVED.get(key)
        if path and os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    content = f.read()
                st.download_button(
                    f"Download {label}",
                    data=content,
                    file_name=os.path.basename(path),
                    key=f"download_{key}"
                )
            except Exception as e:
                st.warning(f"Could not read {label}: {e}")
        else:
            st.caption(f"Not available: {label}")

    st.markdown("---")
    st.markdown("### Generate project summary PDF")

    stats_json, stats_err = safe_json("statistics_json")
    scenario_df, scenario_err = safe_csv("scenario_csv")
    stress_arr, stress_meta, stress_err = safe_raster("stress")
    priority_arr, priority_meta, priority_err = safe_raster("priority")

    summary = {
        "Project": "TerraVita Bayad Agricultural Digital Twin",
        "Study area": "Bayad Taluka, Aravalli, Gujarat",
        "Project directory": PROJECT,
        "Boundary available": boundary is not None,
        "Satellite indices available": RESOLVED["indices"] is not None,
        "Agricultural stress available": stress_arr is not None,
        "GeoAI clusters available": RESOLVED["clusters"] is not None,
        "Intervention priority available": priority_arr is not None,
    }

    if boundary is not None:
        try:
            summary["Boundary area (km²)"] = round(
                boundary.to_crs("EPSG:32643").geometry.area.sum() / 1e6,
                2
            )
        except Exception:
            pass

    if stress_arr is not None:
        stress_stats = finite_stats(stress_arr)
        summary["Mean relative stress"] = round(stress_stats["mean"], 4)

    if priority_arr is not None:
        priority_stats = finite_stats(priority_arr)
        summary["Mean priority score"] = round(priority_stats["mean"], 4)

    if stats_json:
        summary["Candidate count"] = stats_json.get("candidate_count", "N/A")

    pdf_bytes = make_pdf_report(summary, scenario_df)

    st.download_button(
        "Generate and download PDF summary",
        data=pdf_bytes,
        file_name="TerraVita_Bayad_Digital_Twin_Summary.pdf",
        mime="application/pdf"
    )

    # Build a ZIP from available smaller outputs and report tables.
    if st.button("Prepare project report package ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for key in [
                "candidates_csv",
                "scenario_csv",
                "statistics_json"
            ]:
                path = RESOLVED.get(key)
                if path and os.path.isfile(path):
                    zf.write(path, arcname=os.path.basename(path))

            zf.writestr(
                "TerraVita_Dashboard_Summary.json",
                json.dumps(summary, indent=2)
            )

            zf.writestr(
                "README_LIMITATIONS.txt",
                "TerraVita Bayad Agricultural Digital Twin dashboard package.\n"
                "Scenario coverage is spatial buffer overlap only.\n"
                "It does not estimate groundwater recharge, crop yield, "
                "water savings or intervention effectiveness.\n"
                "Candidate points require field and engineering validation.\n"
            )

        zip_buffer.seek(0)
        st.download_button(
            "Download report package ZIP",
            data=zip_buffer.getvalue(),
            file_name="TerraVita_Bayad_Report_Package.zip",
            mime="application/zip"
        )

# ============================================================
# 16. DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.markdown('<div class="tv-section">08 / Data validation</div>', unsafe_allow_html=True)
    st.header("Data Availability & Quality Checks")

    rows = []

    for key, paths in PATHS.items():
        resolved = RESOLVED.get(key)
        row = {
            "Dataset": key,
            "Status": "Available" if resolved else "Missing",
            "Resolved path": resolved if resolved else " / ".join(paths)
        }

        if resolved and resolved.lower().endswith((".tif", ".tiff")):
            try:
                with rasterio.open(resolved) as src:
                    row["CRS"] = str(src.crs)
                    row["Width"] = src.width
                    row["Height"] = src.height
                    row["Bands"] = src.count
                    row["NoData"] = src.nodata
                    row["Data type"] = ", ".join(src.dtypes)
                    row["Validation"] = (
                        "OK" if src.crs is not None and src.count > 0
                        else "Check CRS / bands"
                    )
            except Exception as e:
                row["Validation"] = f"ERROR: {e}"

        elif resolved:
            row["Validation"] = "File found; vector/table content checked when loaded."

        rows.append(row)

    quality_df = pd.DataFrame(rows)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.markdown("### Important band assumptions")
    st.write(
        "- Module 02 index stack: Band 1 NDVI, Band 2 NDWI, "
        "Band 3 MNDWI, Band 4 NDMI, Band 5 valid observations."
    )
    st.write(
        "- Module 03 terrain stack: this dashboard assumes Band 1 elevation "
        "and Band 2 slope for the map selector. Confirm the actual export order."
    )
    st.write(
        "- Module 04 cluster raster: cluster IDs are categorical labels."
    )
    st.write(
        "- Module 05 priority raster: expected continuous score between 0 and 1."
    )

    st.markdown(
        """
        <div class="tv-note">
        A file being present does not prove its scientific validity. Confirm
        acquisition dates, CRS, band order, nodata handling, spatial alignment,
        source provenance and administrative boundary vintage before using
        the dashboard for client or policy decisions.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 17. FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#829EAA;font-size:11px;padding:8px;">
      TERRAVITA ENVIRONMENTAL & GEOSPATIAL CONSULTANCY
      <br>
      Satellite + GIS + AI for Environmental, Urban & Geological Decision-Making
      <br>
      Bayad Agricultural Digital Twin · Prototype Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
