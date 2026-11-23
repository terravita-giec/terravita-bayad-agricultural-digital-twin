
from pathlib import Path
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from rasterio.plot import plotting_extent

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# TERRAVITA - BAYAD AGRICULTURAL DIGITAL TWIN
# =========================================================

st.set_page_config(
    page_title="TerraVita | Bayad Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

FILES = {
    "boundary": DATA / "Bayad_Taluka_2025.geojson",

    "indices": DATA / "module02" /
        "Bayad_Kharif_2025_Indices.tif",

    "terrain": DATA / "modul03" /
        "Bayad_SRTM_Terrain_30m.tif",

    "rainfall": DATA / "modul03" /
        "Bayad_Kharif_2025_Rainfall.tif",

    "streams": DATA / "modul03" /
        "Bayad_Streams_1000_cells.tif",

    "stress": DATA / "model_04" /
        "Bayad_Kharif2025_Relative_Stress.tif",

    "stress_classes": DATA / "model_04" /
        "Bayad_Kharif2025_Stress_Classes.tif",

    "drainage": DATA / "model_04" /
        "Bayad_Kharif2025_Drainage_Distance.tif",

    "clusters": DATA / "model_04" /
        "Bayad_Kharif2025_GeoAI_Clusters.tif",

    "priority": DATA / "module05" /
        "Bayad_Digital_Twin_Priority_Score.tif",

    "candidates": DATA / "module05" /
        "Bayad_Intervention_Candidates.gpkg",
}


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #08131e;
    color: #e8f0f4;
}
[data-testid="stSidebar"] {
    background-color: #0b1925;
}
h1, h2, h3 {
    color: #e8f0f4;
}
.metric-card {
    background: #102333;
    border: 1px solid #20394a;
    padding: 18px;
    border-radius: 10px;
}
.small-muted {
    color: #8fa5b3;
    font-size: 12px;
}
div[data-testid="stMetric"] {
    background: #102333;
    border: 1px solid #20394a;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA VALIDATION
# =========================================================

def file_status(path):
    return path.exists() and path.is_file()


def read_raster(path):
    """
    Read a GeoTIFF safely.
    Returns dataset array and metadata.
    """
    if not file_status(path):
        return None, None, f"File not found: {path}"

    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                return None, None, "Raster contains no bands."

            array = src.read(masked=True)
            array = np.ma.filled(array, np.nan).astype("float32")

            metadata = {
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "dtype": src.dtypes[0]
            }

            return array, metadata, None

    except Exception as exc:
        return None, None, str(exc)


def read_vector(path):
    if not file_status(path):
        return None, f"File not found: {path}"

    try:
        gdf = gpd.read_file(path)

        if gdf.empty:
            return None, "Vector file contains no features."

        if gdf.crs is None:
            return None, "Vector file has no CRS defined."

        gdf = gdf[gdf.geometry.notna()].copy()

        if gdf.empty:
            return None, "Vector file contains no valid geometries."

        return gdf, None

    except Exception as exc:
        return None, str(exc)


def raster_summary(array):
    values = array[np.isfinite(array)]

    if values.size == 0:
        return None

    return {
        "valid_pixels": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values))
    }


def show_raster(array, metadata, band_index, title, cmap="viridis"):
    """
    Display a raster band and optionally overlay the study boundary.
    """
    band = array[band_index]

    fig, ax = plt.subplots(figsize=(10, 7))

    extent = (
        metadata["bounds"].left,
        metadata["bounds"].right,
        metadata["bounds"].bottom,
        metadata["bounds"].top
    )

    image = ax.imshow(
        band,
        extent=extent,
        origin="upper",
        cmap=cmap
    )

    boundary, error = read_vector(FILES["boundary"])

    if boundary is not None:
        try:
            with rasterio.open_for_reading if False else open_vector_context():
                pass
        except Exception:
            pass

        try:
            raster_crs = metadata["crs"]

            if raster_crs and boundary.crs:
                boundary_plot = boundary.to_crs(raster_crs)
                boundary_plot.boundary.plot(
                    ax=ax,
                    color="#18b6a4",
                    linewidth=1.2
                )
        except Exception:
            pass

    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#102333")
    ax.set_facecolor("#102333")

    colorbar = fig.colorbar(image, ax=ax, shrink=0.75)
    colorbar.ax.tick_params(colors="white")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def open_vector_context():
    # Placeholder context manager is not used for data access.
    from contextlib import nullcontext
    return nullcontext()


# =========================================================
# HEADER
# =========================================================

st.title("🌍 TERRAVITA")
st.subheader("Bayad Agricultural Digital Twin")

st.markdown(
    "Satellite-based agricultural monitoring, spatial analysis "
    "and watershed resilience decision support."
)

st.caption(
    "Bayad Taluka | Aravalli District | Gujarat, India | "
    "Kharif 2025 analytical prototype"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("TERRAVITA")
st.sidebar.caption("GEOSPATIAL INTELLIGENCE")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "GIS Map",
        "Satellite Indicators",
        "Agricultural Stress",
        "Rainfall & Terrain",
        "GeoAI Clusters",
        "Digital Twin Scenarios",
        "Data Quality"
    ]
)

st.sidebar.divider()
st.sidebar.caption("PROJECT STATUS")

for key, path in FILES.items():
    if key == "candidates":
        continue

    status = "Available" if file_status(path) else "Missing"

    st.sidebar.write(
        f"{'🟢' if status == 'Available' else '🔴'} "
        f"{key.replace('_', ' ').title()}"
    )


# =========================================================
# LOADERS
# =========================================================

@st.cache_data(show_spinner=False)
def cached_raster(path_string):
    array, metadata, error = read_raster(Path(path_string))
    return array, metadata, error


@st.cache_data(show_spinner=False)
def cached_vector(path_string):
    gdf, error = read_vector(Path(path_string))

    if gdf is None:
        return None, error

    return gdf.to_json(), None


def get_raster(key):
    return cached_raster(str(FILES[key]))


def get_boundary():
    geojson, error = cached_vector(str(FILES["boundary"]))

    if geojson is None:
        return None, error

    return gpd.read_file(FILES["boundary"]), None


def raster_panel(key, title, cmap="viridis"):
    array, metadata, error = get_raster(key)

    if error:
        st.warning(error)
        return

    band_options = [
        f"Band {i + 1}" for i in range(array.shape[0])
    ]

    selected_band = st.selectbox(
        f"{title} band",
        band_options,
        key=f"{key}_band"
    )

    band_index = int(selected_band.split()[1]) - 1

    show_raster(
        array,
        metadata,
        band_index,
        title,
        cmap
    )

    summary = raster_summary(array[band_index])

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Minimum", f"{summary['minimum']:.3f}")
        c2.metric("Maximum", f"{summary['maximum']:.3f}")
        c3.metric("Mean", f"{summary['mean']:.3f}")

        st.caption(
            f"Valid pixels: {summary['valid_pixels']:,} | "
            f"CRS: {metadata['crs']} | "
            f"Resolution: {metadata['width']} × {metadata['height']}"
        )


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Regional Overview")

    boundary, boundary_error = get_boundary()

    area_km2 = None

    if boundary is not None:
        try:
            area_km2 = (
                boundary.to_crs("EPSG:32643").geometry.area.sum()
                / 1_000_000
            )
        except Exception:
            pass

    stress_array, stress_meta, stress_error = get_raster("stress")

    stress_mean = None

    if stress_array is not None:
        stats = raster_summary(stress_array[0])
        if stats:
            stress_mean = stats["mean"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Study Area",
        f"{area_km2:.2f} km²" if area_km2 is not None else "Unavailable"
    )

    c2.metric(
        "Monitoring",
        "2021–2025"
    )

    c3.metric(
        "Mean Relative Stress",
        f"{stress_mean:.3f}" if stress_mean is not None else "Unavailable"
    )

    c4.metric(
        "Satellite",
        "Sentinel-2"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Agricultural Stress")
        if stress_array is not None:
            show_raster(
                stress_array,
                stress_meta,
                0,
                "Relative Agricultural Stress",
                "RdYlGn_r"
            )
        else:
            st.info(stress_error)

    with right:
        st.subheader("Digital Twin Priority")

        priority_array, priority_meta, priority_error = get_raster("priority")

        if priority_array is not None:
            show_raster(
                priority_array,
                priority_meta,
                0,
                "Intervention Priority Score",
                "magma"
            )
        else:
            st.info(priority_error)

    st.info(
        "Interpretation: relative stress and priority scores are "
        "screening outputs. They are not direct measurements of crop "
        "yield, groundwater recharge or intervention benefits."
    )


# =========================================================
# GIS MAP
# =========================================================

elif page == "GIS Map":

    st.header("GIS Study Area")

    boundary, error = get_boundary()

    if error:
        st.error(error)
    else:
        st.write("Study-area boundary loaded successfully.")

        st.map(
            boundary.to_crs("EPSG:4326").get_coordinates()
            if False else pd.DataFrame({
                "latitude": boundary.to_crs("EPSG:4326")
                    .geometry.representative_point().y,
                "longitude": boundary.to_crs("EPSG:4326")
                    .geometry.representative_point().x
            }),
            latitude="latitude",
            longitude="longitude"
        )

        st.write("Boundary attributes")
        st.dataframe(
            boundary.drop(columns="geometry"),
            use_container_width=True
        )

        st.download_button(
            "Download boundary GeoJSON",
            data=boundary.to_json(),
            file_name="Bayad_Taluka_2025.geojson",
            mime="application/geo+json"
        )

    st.caption(
        "For detailed raster overlays and interactive GIS layers, "
        "connect the exported GeoTIFFs and a web map component."
    )


# =========================================================
# SATELLITE INDICATORS
# =========================================================

elif page == "Satellite Indicators":

    st.header("Satellite Indicators")

    st.write(
        "Sentinel-2 seasonal composite. Expected band order: "
        "NDVI, NDWI, MNDWI, NDMI, Valid Observations."
    )

    raster_panel(
        "indices",
        "Kharif 2025 Satellite Indices",
        "RdYlGn"
    )

    st.info(
        "NDVI is a vegetation greenness indicator. NDMI is related "
        "to vegetation moisture response. NDWI and MNDWI support "
        "water-feature screening. None is a direct crop-yield measure."
    )


# =========================================================
# AGRICULTURAL STRESS
# =========================================================

elif page == "Agricultural Stress":

    st.header("Agricultural Stress Assessment")

    raster_panel(
        "stress",
        "Relative Agricultural Stress",
        "RdYlGn_r"
    )

    st.divider()

    raster_panel(
        "stress_classes",
        "Stress Classification",
        "RdYlGn_r"
    )

    st.caption(
        "Stress categories are relative to the selected study-period "
        "data distribution, not validated drought thresholds."
    )


# =========================================================
# RAINFALL & TERRAIN
# =========================================================

elif page == "Rainfall & Terrain":

    st.header("Rainfall and Terrain Intelligence")

    st.subheader("Rainfall")
    raster_panel(
        "rainfall",
        "Rainfall Analysis",
        "BrBG"
    )

    st.divider()

    st.subheader("Terrain")
    raster_panel(
        "terrain",
        "SRTM Terrain",
        "terrain"
    )

    st.divider()

    st.subheader("Drainage Distance")
    raster_panel(
        "drainage",
        "Distance from Extracted Drainage",
        "Blues"
    )

    st.caption(
        "CHIRPS rainfall has a much coarser spatial resolution than "
        "the terrain and Sentinel-2 rasters. Displayed pixel size "
        "does not imply finer rainfall information."
    )


# =========================================================
# GEOAI CLUSTERS
# =========================================================

elif page == "GeoAI Clusters":

    st.header("GeoAI Agricultural Clusters")

    raster_panel(
        "clusters",
        "K-Means Spatial Clusters",
        "tab10"
    )

    st.info(
        "K-Means clusters are exploratory spatial groups. They are "
        "not ground-truth crop types or independently validated "
        "agricultural classes."
    )


# =========================================================
# DIGITAL TWIN SCENARIOS
# =========================================================

elif page == "Digital Twin Scenarios":

    st.header("Digital Twin Intervention Priorities")

    priority_array, priority_meta, priority_error = get_raster("priority")

    if priority_array is not None:

        show_raster(
            priority_array,
            priority_meta,
            0,
            "Digital Twin Priority Score",
            "magma"
        )

        stats = raster_summary(priority_array[0])

        if stats:
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Score", f"{stats['mean']:.3f}")
            c2.metric("Minimum", f"{stats['minimum']:.3f}")
            c3.metric("Maximum", f"{stats['maximum']:.3f}")

    else:
        st.warning(priority_error)

    st.divider()

    st.subheader("Intervention Candidate Locations")

    candidate_path = FILES["candidates"]

    if file_status(candidate_path):

        candidates, error = read_vector(candidate_path)

        if error:
            st.error(error)
        else:
            st.write(f"Candidate features: {len(candidates)}")

            st.dataframe(
                candidates.drop(columns="geometry"),
                use_container_width=True
            )

            st.download_button(
                "Download candidate locations",
                data=candidates.to_csv(index=False),
                file_name="Bayad_Intervention_Candidates.csv",
                mime="text/csv"
            )

    else:
        st.warning(
            "Candidate GeoPackage not found. Add the Module 05 "
            "candidate output to data/module05/."
        )

    st.warning(
        "Priority scores and candidate locations are preliminary "
        "screening outputs. They do not establish engineering "
        "suitability, water availability, recharge, yield improvement "
        "or construction feasibility."
    )


# =========================================================
# DATA QUALITY
# =========================================================

elif page == "Data Quality":

    st.header("Project Data Availability")

    rows = []

    for key, path in FILES.items():

        exists = file_status(path)

        row = {
            "Dataset": key,
            "Expected file": str(path.relative_to(ROOT)),
            "Status": "Available" if exists else "Missing"
        }

        if exists and path.suffix.lower() in [".tif", ".tiff"]:
            _, metadata, error = get_raster(key)

            if metadata:
                row["CRS"] = metadata["crs"]
                row["Bands"] = metadata["bands"]
                row["Dimensions"] = (
                    f"{metadata['width']} × {metadata['height']}"
                )
            else:
                row["Read error"] = error

        rows.append(row)

    inventory = pd.DataFrame(rows)

    st.dataframe(
        inventory,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download data inventory CSV",
        data=inventory.to_csv(index=False),
        file_name="TerraVita_Data_Inventory.csv",
        mime="text/csv"
    )

    missing = int((inventory["Status"] == "Missing").sum())

    if missing == 0:
        st.success("All configured datasets are available.")
    else:
        st.warning(f"{missing} configured dataset(s) are missing.")

    st.caption(
        "A file being present does not establish its scientific "
        "validity, correct date, CRS alignment or data quality."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "TERRAVITA | Satellite + GIS + AI | "
    "Bayad Agricultural Digital Twin | Prototype"
)
