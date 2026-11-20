
import os
import io
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import array_bounds

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="TerraVita | Bayad Agricultural Digital Twin",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 TerraVita")
st.subheader("AI-Driven Agricultural Digital Twin")
st.caption(
    "Bayad Taluka | Aravalli District | Gujarat, India | "
    "Kharif 2025 pilot"
)

st.info(
    "Exploratory geospatial decision-support system. "
    "Outputs require field verification and are not engineering "
    "or agronomic prescriptions."
)

# ---------------------------------------------------------
# 2. FILE CONFIGURATION
# Change these paths to match your Google Drive files.
# ---------------------------------------------------------

BASE = "/content/drive/MyDrive/TERRAVITA_BAYAD_DIGITAL_TWIN"

FILES = {
    "indices": f"{BASE}/01_MODULE_02/Bayad_Kharif_2025_Indices.tif",

    "terrain": f"{BASE}/02_MODULE_03/Bayad_SRTM_Terrain_30m.tif",

    "rainfall": f"{BASE}/02_MODULE_03/Bayad_Kharif_2025_Rainfall.tif",

    "streams": f"{BASE}/02_MODULE_03/Bayad_Streams_1000_cells.tif",

    "stress": f"{BASE}/03_MODULE_04/Bayad_Kharif2025_Relative_Stress.tif",

    "stress_classes": f"{BASE}/03_MODULE_04/Bayad_Kharif2025_Stress_Classes.tif",

    "drainage_distance": f"{BASE}/03_MODULE_04/Bayad_Kharif2025_Drainage_Distance.tif",

    "clusters": f"{BASE}/03_MODULE_04/Bayad_Kharif2025_GeoAI_Clusters.tif",

    "priority": f"{BASE}/04_MODULE_05/Bayad_DigitalTwin_Priority_Score.tif",

    "priority_classes": f"{BASE}/04_MODULE_05/Bayad_DigitalTwin_Priority_Classes.tif",

    "candidates": f"{BASE}/04_MODULE_05/Bayad_Intervention_Candidates.gpkg",

    "scenario": f"{BASE}/04_MODULE_05/Bayad_Scenario_Coverage.csv"
}

# ---------------------------------------------------------
# 3. FILE VALIDATION
# ---------------------------------------------------------

st.sidebar.header("Project data status")

missing = []

for key, path in FILES.items():
    exists = os.path.exists(path)

    if exists:
        st.sidebar.success(f"{key}: available")
    else:
        st.sidebar.error(f"{key}: missing")
        missing.append(key)

if missing:
    st.warning(
        "Some project files are missing. Check the paths in "
        "the FILES dictionary. Available layers can still be explored."
    )

# ---------------------------------------------------------
# 4. RASTER LOADER
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raster(path, band=1, max_dimension=1200):

    with rasterio.open(path) as src:

        scale = min(
            1,
            max_dimension / max(src.width, src.height)
        )

        out_height = max(1, int(src.height * scale))
        out_width = max(1, int(src.width * scale))

        arr = src.read(
            band,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest
        ).astype("float32")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        bounds = src.bounds
        crs = src.crs

        transform = src.transform * src.transform.scale(
            src.width / out_width,
            src.height / out_height
        )

        return arr, bounds, crs, transform


def raster_available(key):
    return os.path.exists(FILES[key])


def show_raster(key, title, band=1, cmap="viridis"):

    if not raster_available(key):
        st.warning(f"{title}: raster file not found.")
        return

    arr, bounds, crs, transform = load_raster(
        FILES[key], band=band
    )

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        st.warning(f"{title}: no valid pixels.")
        return

    fig = px.imshow(
        arr,
        color_continuous_scale=cmap,
        aspect="equal",
        origin="upper",
        title=title
    )

    fig.update_layout(
        height=550,
        coloraxis_colorbar_title=title
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"CRS: {crs} | Valid pixels: {finite.size:,} | "
        f"Minimum: {np.nanmin(finite):.3f} | "
        f"Maximum: {np.nanmax(finite):.3f}"
    )

# ---------------------------------------------------------
# 5. LOAD CANDIDATES AND SCENARIOS
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_candidates(path):

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise ValueError("Candidate layer has no CRS.")

    return gdf


@st.cache_data(show_spinner=False)
def load_scenario(path):

    return pd.read_csv(path)


# ---------------------------------------------------------
# 6. SIDEBAR NAVIGATION
# ---------------------------------------------------------

section = st.sidebar.radio(
    "Explore dashboard",
    [
        "Overview",
        "Satellite Indicators",
        "Agricultural Stress",
        "Rainfall & Terrain",
        "Intervention Priorities",
        "Scenario Analysis",
        "Data & Methodology"
    ]
)

# ---------------------------------------------------------
# 7. OVERVIEW
# ---------------------------------------------------------

if section == "Overview":

    st.header("Bayad Agricultural Landscape — Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Study area", "Bayad Taluka")
    col2.metric("Pilot season", "Kharif 2025")
    col3.metric("Platform", "GIS + Remote Sensing + GeoAI")

    st.markdown("### Digital twin workflow")

    st.markdown("""
    **Satellite observations**
    → **Environmental indicators**
    → **Agricultural stress screening**
    → **Spatial priority assessment**
    → **Field investigation planning**
    """)

    if raster_available("priority"):
        show_raster(
            "priority",
            "Preliminary Intervention Priority Score",
            cmap="YlOrRd"
        )
    else:
        st.warning("Priority raster is not available yet.")

# ---------------------------------------------------------
# 8. SATELLITE INDICATORS
# ---------------------------------------------------------

elif section == "Satellite Indicators":

    st.header("Satellite-Based Agricultural Indicators")

    indicator = st.selectbox(
        "Select indicator",
        ["NDVI", "NDMI", "NDWI", "MNDWI"]
    )

    band_map = {
        "NDVI": 1,
        "NDWI": 2,
        "MNDWI": 3,
        "NDMI": 4
    }

    show_raster(
        "indices",
        f"{indicator} — Kharif 2025",
        band=band_map[indicator],
        cmap="RdYlGn"
    )

    st.caption(
        "NDVI indicates vegetation greenness; NDMI is a vegetation "
        "moisture-related spectral indicator. These indices are not "
        "direct measurements of crop yield, soil moisture or groundwater."
    )

# ---------------------------------------------------------
# 9. AGRICULTURAL STRESS
# ---------------------------------------------------------

elif section == "Agricultural Stress":

    st.header("Relative Agricultural Stress Screening")

    col1, col2 = st.columns(2)

    with col1:
        show_raster(
            "stress",
            "Relative Vegetation Stress",
            cmap="YlOrRd"
        )

    with col2:
        show_raster(
            "stress_classes",
            "Relative Stress Classes",
            cmap="RdYlGn_r"
        )

    st.markdown("""
    **Interpretation**

    - Lower relative stress: lower values within the selected pilot area.
    - Moderate relative stress: intermediate values.
    - Elevated or higher relative stress: higher values within the
      selected pilot area.

    These are relative screening classes, not calibrated drought
    or crop-damage thresholds.
    """)

    if raster_available("clusters"):
        show_raster(
            "clusters",
            "Exploratory GeoAI Clusters",
            cmap="Set2"
        )

        st.caption(
            "Unsupervised clusters describe spectral and terrain similarity. "
            "They are not validated crop or land-cover classes."
        )

# ---------------------------------------------------------
# 10. RAINFALL AND TERRAIN
# ---------------------------------------------------------

elif section == "Rainfall & Terrain":

    st.header("Rainfall, Terrain and Drainage")

    tab1, tab2, tab3 = st.tabs([
        "Rainfall",
        "Terrain",
        "Drainage"
    ])

    with tab1:
        show_raster(
            "rainfall",
            "Kharif Rainfall / Anomaly Raster",
            cmap="BrBG"
        )

        st.caption(
            "Check the rainfall GeoTIFF band description and units "
            "before interpreting the displayed values."
        )

    with tab2:
        show_raster(
            "terrain",
            "Terrain Elevation",
            band=1,
            cmap="terrain"
        )

        show_raster(
            "terrain",
            "Terrain Slope",
            band=2,
            cmap="YlOrBr"
        )

    with tab3:
        show_raster(
            "drainage_distance",
            "Distance to Derived Drainage Network",
            cmap="Blues_r"
        )

        show_raster(
            "streams",
            "Derived Stream Network",
            cmap="Blues"
        )

        st.caption(
            "Drainage lines are DEM-derived exploratory features, "
            "not surveyed or officially mapped watercourses."
        )

# ---------------------------------------------------------
# 11. INTERVENTION PRIORITIES
# ---------------------------------------------------------

elif section == "Intervention Priorities":

    st.header("Preliminary Intervention Priority Assessment")

    show_raster(
        "priority",
        "Priority Score",
        cmap="YlOrRd"
    )

    show_raster(
        "priority_classes",
        "Priority Classes",
        cmap="RdYlGn_r"
    )

    if raster_available("candidates"):

        try:
            candidates = load_candidates(FILES["candidates"])

            st.markdown("### Candidate field-investigation locations")

            st.write(f"Candidate locations: {len(candidates)}")

            st.dataframe(
                candidates.drop(columns="geometry", errors="ignore"),
                use_container_width=True
            )

            st.download_button(
                "Download candidate locations (CSV)",
                candidates.drop(
                    columns="geometry",
                    errors="ignore"
                ).to_csv(index=False).encode("utf-8"),
                file_name="TerraVita_Bayad_Candidates.csv",
                mime="text/csv"
            )

            st.caption(
                "These are locations for field investigation, not "
                "approved sites for constructing recharge or water "
                "management structures."
            )

        except Exception as e:
            st.error(f"Could not read candidate layer: {e}")

    else:
        st.warning("Candidate GeoPackage not found.")

# ---------------------------------------------------------
# 12. SCENARIO ANALYSIS
# ---------------------------------------------------------

elif section == "Scenario Analysis":

    st.header("Intervention Buffer Scenario Analysis")

    if raster_available("scenario"):

        try:
            scenario = load_scenario(FILES["scenario"])

            st.dataframe(
                scenario,
                use_container_width=True
            )

            numeric_columns = scenario.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_columns:
                selected_metric = st.selectbox(
                    "Select numerical metric",
                    numeric_columns
                )

                category_columns = [
                    c for c in scenario.columns
                    if c not in numeric_columns
                ]

                if category_columns:
                    selected_category = st.selectbox(
                        "Group by",
                        category_columns
                    )

                    fig = px.bar(
                        scenario,
                        x=selected_category,
                        y=selected_metric,
                        title=f"{selected_metric} by {selected_category}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            st.download_button(
                "Download scenario results",
                scenario.to_csv(index=False).encode("utf-8"),
                file_name="TerraVita_Bayad_Scenario_Results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Could not read scenario CSV: {e}")

    else:
        st.warning("Scenario coverage CSV not found.")

    st.caption(
        "Buffer coverage measures spatial overlap with selected "
        "priority pixels. It does not measure water capture, recharge, "
        "yield improvement or actual intervention effectiveness."
    )

# ---------------------------------------------------------
# 13. DATA AND METHODOLOGY
# ---------------------------------------------------------

elif section == "Data & Methodology":

    st.header("Data Sources and Methodology")

    methodology = pd.DataFrame({
        "Dataset": [
            "Sentinel-2 Surface Reflectance",
            "CHIRPS Daily Rainfall",
            "SRTM Digital Elevation Model",
            "DEM-derived drainage",
            "Vegetation stress screening",
            "KMeans exploratory clusters",
            "Priority scoring",
            "Intervention buffer scenarios"
        ],
        "Role": [
            "Satellite spectral indicators",
            "Rainfall totals and anomalies",
            "Elevation and slope",
            "Terrain-derived drainage proxy",
            "Relative vegetation stress",
            "Unsupervised pattern exploration",
            "Weighted spatial screening",
            "Priority-pixel buffer coverage"
        ]
    })

    st.dataframe(methodology, use_container_width=True)

    st.markdown("### Key limitations")

    st.markdown("""
    - Satellite indicators are indirect observations.
    - Rainfall data are much coarser than Sentinel-2 imagery.
    - Terrain-derived drainage is not a verified stream survey.
    - Stress classes are relative to the study area's distribution.
    - GeoAI clusters have not been validated with field labels.
    - Priority weights are exploratory assumptions.
    - Candidate locations require field, land ownership,
      soil, geology and hydrogeological assessment.
    """)

# ---------------------------------------------------------
# 14. FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "TerraVita Environmental & Geospatial Consultancy | "
    "Bayad Agricultural Digital Twin | Prototype"
)

