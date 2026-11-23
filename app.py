"""
TerraVita | Bayad Agricultural Digital Twin - Module 06 dashboard
Runs on Streamlit Community Cloud or locally (streamlit run app.py).

All paths are relative to this file (the GitHub repo root). The app scans the
`data/` and `outputs/` folders for your Module 01-05 files by name pattern, so
you do not need to hard-code every path. If a file is not found by pattern,
set an exact path in OVERRIDES below.
"""

import fnmatch
import io
from datetime import date
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from streamlit_folium import st_folium

st.set_page_config(
    page_title="TerraVita | Bayad Digital Twin",
    page_icon="🌍",
    layout="wide",
)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [BASE_DIR / "data", BASE_DIR / "outputs"]
PROJECT_CRS = "EPSG:32643"          # UTM 43N, suitable for Bayad, Gujarat
MAX_DIM = 900                        # max preview raster size in pixels
DEFAULT_CENTER = [23.25, 73.30]      # fallback map centre (Bayad area)
ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Exact paths (relative to repo root) if automatic search picks the wrong file.
# Example: {"NDVI": "outputs/rasters/bayad_ndvi_2025.tif"}
OVERRIDES = {}

RASTERS = {
    "NDVI": dict(patterns=["*ndvi*.tif", "*ndvi*.tiff"], cmap="RdYlGn"),
    "NDWI": dict(patterns=["*ndwi*.tif", "*ndwi*.tiff"], exclude=["mndwi"], cmap="Blues"),
    "MNDWI": dict(patterns=["*mndwi*.tif", "*mndwi*.tiff"], cmap="GnBu"),
    "NDMI": dict(patterns=["*ndmi*.tif", "*ndmi*.tiff"], cmap="YlGnBu"),
    "Agricultural Stress": dict(patterns=["*stress*.tif", "*stress*.tiff"], cmap="YlOrRd"),
    "GeoAI Clusters": dict(
        patterns=["*cluster*.tif", "*kmeans*.tif", "*geoai*.tif"], cmap="tab10", categorical=True
    ),
    "Intervention Priority": dict(patterns=["*priority*.tif"], cmap="magma"),
    "Elevation": dict(patterns=["*srtm*.tif", "*dem*.tif", "*elevation*.tif"], cmap="terrain"),
    "Slope": dict(patterns=["*slope*.tif"], cmap="viridis"),
    "Rainfall": dict(patterns=["*chirps*.tif", "*rainfall*.tif", "*rain*.tif"], cmap="Blues"),
}

VECTORS = {
    "AOI Boundary": ["*boundary*.gpkg", "*aoi*.gpkg", "*boundary*.geojson", "*aoi*.geojson",
                     "*boundary*.shp", "*aoi*.shp", "*bayad*.gpkg"],
    "Intervention Candidates": ["*candidate*.gpkg", "*candidate*.geojson", "*candidate*.shp",
                                "*candidate*.csv"],
    "Scenario Buffers": ["*buffer*.gpkg", "*buffer*.geojson", "*scenario*.gpkg"],
}

TABLES = {
    "Scenario Summary": ["*scenario*summary*.csv", "*scenario*.csv", "*coverage*.csv"],
    "Stress Summary": ["*stress*summary*.csv", "*stress*.csv"],
    "Cluster Summary": ["*cluster*summary*.csv", "*cluster*.csv"],
}

INDEX_NAMES = ["NDVI", "NDWI", "MNDWI", "NDMI"]

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp { background:#081522; color:#edf5f7; }
section[data-testid="stSidebar"] { background:#0b1d2b; border-right:1px solid #244657; }
div[data-testid="stMetric"] { background:#102738; border:1px solid #244657;
    padding:14px 16px; border-radius:12px; }
div[data-testid="stMetricLabel"] { color:#a8bdc7; }
div[data-testid="stMetricValue"] { color:#35c6b5; }
h1, h2, h3 { color:#edf5f7; }
.tv-hero { background:linear-gradient(135deg,#0a1c2b,#102f40); border:1px solid #244657;
    border-radius:14px; padding:26px 30px; margin-bottom:20px; }
.tv-hero .eyebrow { color:#35c6b5; font-size:12px; letter-spacing:3px; font-weight:bold; }
.tv-hero h1 { margin:8px 0 6px; font-size:30px; }
.tv-hero p { color:#a8bdc7; margin:0; }
.tv-note { background:#102738; border-left:4px solid #35c6b5; padding:10px 14px;
    border-radius:6px; color:#a8bdc7; font-size:14px; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# FILE DISCOVERY
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_files():
    files = []
    for d in SEARCH_DIRS:
        if d.exists():
            files += [p for p in d.rglob("*") if p.is_file()]
    return sorted(files)


def locate(name, patterns, exclude=()):
    override = OVERRIDES.get(name)
    if override:
        p = BASE_DIR / override
        return p if p.exists() else None
    files = list_files()
    for pat in patterns:
        for p in files:
            n = p.name.lower()
            if any(e in n for e in exclude):
                continue
            if fnmatch.fnmatch(n, pat):
                return p
    return None


@st.cache_data(show_spinner=False)
def discover():
    found = {}
    for n, c in RASTERS.items():
        found[n] = locate(n, c["patterns"], c.get("exclude", ()))
    for n, pats in VECTORS.items():
        found[n] = locate(n, pats)
    for n, pats in TABLES.items():
        found[n] = locate(n, pats)
    return found


ds = discover()


def rel(p):
    try:
        return str(p.relative_to(BASE_DIR))
    except Exception:
        return str(p)


# ----------------------------------------------------------------------------
# RASTER HELPERS (validation, reading, rendering)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def validate_raster(path_str):
    info = {"ok": False, "errors": [], "warnings": []}
    try:
        with rasterio.open(path_str) as src:
            info.update(
                width=src.width, height=src.height, bands=src.count,
                crs=str(src.crs) if src.crs else None, nodata=src.nodata,
                dtype=src.dtypes[0],
            )
            if src.crs is None:
                info["errors"].append("No CRS defined")
            if src.count < 1 or src.width < 1 or src.height < 1:
                info["errors"].append("Empty raster")
            if src.count > 1:
                info["warnings"].append(f"{src.count} bands found; band 1 is used")
            if src.nodata is None:
                info["warnings"].append("No nodata value set")
            if not info["errors"]:
                step = max(1, int(max(src.width, src.height) / 600))
                arr = src.read(
                    1,
                    out_shape=(max(1, src.height // step), max(1, src.width // step)),
                    masked=True,
                )
                data = np.ma.filled(arr.astype("float64"), np.nan)
                finite = data[np.isfinite(data)]
                info["valid_pct"] = round(100 * finite.size / data.size, 1)
                if finite.size == 0:
                    info["errors"].append("No valid pixels (all nodata/NaN)")
                else:
                    info["min"], info["max"] = float(finite.min()), float(finite.max())
                    info["mean"] = float(finite.mean())
                    if finite.min() == finite.max():
                        info["warnings"].append("All valid pixels have the same value")
                    if info["valid_pct"] < 20:
                        info["warnings"].append("Less than 20% valid pixels")
    except Exception as e:
        info["errors"].append(f"Cannot open raster: {e}")
    info["ok"] = not info["errors"]
    return info


@st.cache_data(show_spinner="Reading raster...")
def read_raster(path_str, categorical=False):
    res = Resampling.nearest if categorical else Resampling.bilinear
    with rasterio.open(path_str) as src:
        with WarpedVRT(src, crs="EPSG:4326", resampling=res) as vrt:
            scale = max(1.0, max(vrt.width, vrt.height) / MAX_DIM)
            h, w = max(1, int(vrt.height / scale)), max(1, int(vrt.width / scale))
            arr = vrt.read(1, out_shape=(h, w), masked=True, resampling=res)
            b = vrt.bounds
    data = np.ma.filled(arr.astype("float64"), np.nan)
    return data, (float(b.left), float(b.bottom), float(b.right), float(b.top))


def get_raster(name, quiet=False):
    p = ds.get(name)
    if p is None:
        if not quiet:
            st.warning(f"**{name}**: file not found in `data/` or `outputs/`.")
        return None
    v = validate_raster(str(p))
    if not v["ok"]:
        if not quiet:
            st.error(f"**{name}** (`{rel(p)}`): " + "; ".join(v["errors"]))
        return None
    try:
        return read_raster(str(p), RASTERS[name].get("categorical", False))
    except Exception as e:
        if not quiet:
            st.error(f"**{name}**: could not read raster ({e})")
        return None


def to_rgba(data, cmap_name, categorical=False):
    valid = np.isfinite(data)
    out = np.zeros(data.shape + (4,), dtype=np.uint8)
    if not valid.any():
        return out
    cm = matplotlib.colormaps[cmap_name]
    if categorical:
        idx = np.where(valid, data, 0).astype(int) % 10
        rgba = cm(idx)
    else:
        lo, hi = np.nanpercentile(data[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-9
        norm = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0, 1)
        rgba = cm(norm)
    out = (rgba * 255).astype(np.uint8)
    out[~valid, 3] = 0
    return out


def raster_figure(name, data, bounds, title=None):
    cfg = RASTERS[name]
    cat = cfg.get("categorical", False)
    left, bottom, right, top = bounds
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#081522")
    ax.set_facecolor("#081522")
    if cat:
        vals = np.unique(data[np.isfinite(data)]).astype(int)
        ax.imshow(to_rgba(data, "tab10", True), extent=(left, right, bottom, top))
        cm = matplotlib.colormaps["tab10"]
        handles = [plt.Rectangle((0, 0), 1, 1, color=cm(int(v) % 10)) for v in vals]
        ax.legend(handles, [f"Cluster {v}" for v in vals], loc="lower right", fontsize=8)
    else:
        lo, hi = np.nanpercentile(data[np.isfinite(data)], [2, 98])
        im = ax.imshow(data, extent=(left, right, bottom, top), cmap=cfg["cmap"],
                       vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(colors="#a8bdc7")
    ax.set_aspect(1 / np.cos(np.radians((top + bottom) / 2)))
    ax.set_title(title or name, color="#edf5f7")
    ax.tick_params(colors="#a8bdc7", labelsize=8)
    ax.set_xlabel("Longitude", color="#a8bdc7")
    ax.set_ylabel("Latitude", color="#a8bdc7")
    return fig


def stats_of(data):
    v = data[np.isfinite(data)]
    if v.size == 0:
        return {}
    return {
        "Min": v.min(), "P25": np.percentile(v, 25), "Median": np.median(v),
        "Mean": v.mean(), "P75": np.percentile(v, 75), "Max": v.max(), "Std": v.std(),
    }


# ----------------------------------------------------------------------------
# VECTOR / TABLE HELPERS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_vector(path_str):
    p = Path(path_str)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        low = {c.lower(): c for c in df.columns}
        lat = next((low[k] for k in ("lat", "latitude", "y") if k in low), None)
        lon = next((low[k] for k in ("lon", "lng", "longitude", "x") if k in low), None)
        if lat is None or lon is None:
            raise ValueError("CSV has no lat/lon columns")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    else:
        gdf = gpd.read_file(p)
    if gdf.empty:
        raise ValueError("Layer is empty")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def get_vector(name, quiet=False):
    p = ds.get(name)
    if p is None:
        if not quiet:
            st.warning(f"**{name}**: file not found in `data/` or `outputs/`.")
        return None
    try:
        return load_vector(str(p))
    except Exception as e:
        if not quiet:
            st.error(f"**{name}** (`{rel(p)}`): {e}")
        return None


@st.cache_data(show_spinner=False)
def load_table(path_str):
    return pd.read_csv(path_str)


def get_table(name, quiet=False):
    p = ds.get(name)
    if p is None:
        if not quiet:
            st.info(f"**{name}**: no CSV found.")
        return None
    try:
        return load_table(str(p))
    except Exception as e:
        if not quiet:
            st.error(f"**{name}** (`{rel(p)}`): {e}")
        return None


def union(geoseries):
    fn = getattr(geoseries, "union_all", None)
    return fn() if fn else geoseries.unary_union


def geom_only(gdf, simplify=None):
    g = gdf[["geometry"]].to_crs("EPSG:4326")
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify)
    return g


# ----------------------------------------------------------------------------
# MAP HELPERS
# ----------------------------------------------------------------------------
def new_map(aoi=None, tiles="OpenStreetMap"):
    center, zoom = DEFAULT_CENTER, 10
    if aoi is not None:
        b = aoi.to_crs("EPSG:4326").total_bounds
        center = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=tiles == "OpenStreetMap").add_to(m)
    folium.TileLayer(tiles=ESRI_URL, attr="Esri World Imagery", name="Satellite (Esri)",
                     show=tiles != "OpenStreetMap").add_to(m)
    if aoi is not None:
        b = aoi.to_crs("EPSG:4326").total_bounds
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
        folium.GeoJson(
            geom_only(aoi, 0.0002).to_json(), name="AOI boundary",
            style_function=lambda x: {"color": "#35c6b5", "weight": 2, "fillOpacity": 0},
        ).add_to(m)
    return m


def add_raster(m, name, opacity=0.75):
    r = get_raster(name)
    if r is None:
        return False
    data, (left, bottom, right, top) = r
    cfg = RASTERS[name]
    rgba = to_rgba(data, cfg["cmap"], cfg.get("categorical", False))
    folium.raster_layers.ImageOverlay(
        image=rgba, bounds=[[bottom, left], [top, right]], opacity=opacity,
        name=name, show=True,
    ).add_to(m)
    return True


def add_points(m, gdf, name="Intervention candidates", limit=500):
    g = gdf.copy()
    g["geometry"] = g.geometry.representative_point()
    g = g.to_crs("EPSG:4326").head(limit)
    cols = [c for c in g.columns if c != "geometry"][:4]
    fg = folium.FeatureGroup(name=name)
    for _, row in g.iterrows():
        tip = "<br>".join(f"{c}: {row[c]}" for c in cols) or name
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x], radius=5, color="#ffd166",
            fill=True, fill_opacity=0.85, tooltip=tip,
        ).add_to(fg)
    fg.add_to(m)


def show_map(m, height=620):
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=height, use_container_width=True, returned_objects=[])


# ----------------------------------------------------------------------------
# KPIs AND REPORTS
# ----------------------------------------------------------------------------
def compute_kpis():
    k = {"Layers found": f"{sum(v is not None for v in ds.values())} of {len(ds)}"}
    aoi = get_vector("AOI Boundary", quiet=True)
    if aoi is not None:
        try:
            k["AOI area (km2)"] = f"{aoi.to_crs(PROJECT_CRS).area.sum() / 1e6:,.1f}"
        except Exception:
            pass
    for nm, label in [("NDVI", "Mean NDVI"), ("Agricultural Stress", "Mean stress score")]:
        r = get_raster(nm, quiet=True)
        if r is not None:
            k[label] = f"{np.nanmean(r[0]):.3f}"
    cand = get_vector("Intervention Candidates", quiet=True)
    if cand is not None:
        k["Intervention candidates"] = str(len(cand))
    return k


def build_pdf(kpis):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "TerraVita | Bayad Agricultural Digital Twin")
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Summary report generated {date.today().isoformat()}")
    y -= 34
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Key indicators")
    y -= 20
    c.setFont("Helvetica", 11)
    for key, val in kpis.items():
        c.drawString(60, y, f"{key}: {val}")
        y -= 18
    y -= 16
    c.setFont("Helvetica-Oblique", 9)
    for line in [
        "Screening-level decision-support prototype. Stress values are relative,",
        "not calibrated crop-loss measurements. Verify boundary vintage, CRS,",
        "raster band order and input dates before client use.",
    ]:
        c.drawString(50, y, line)
        y -= 13
    c.showPage()
    c.save()
    return buf.getvalue()


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
def page_overview():
    st.markdown(
        """<div class="tv-hero"><div class="eyebrow">GEOSPATIAL INTELLIGENCE ·
        AGRICULTURAL RESILIENCE</div><h1>Bayad Agricultural Digital Twin</h1>
        <p>GIS, Sentinel-2 remote sensing and GeoAI decision-support prototype for
        Bayad Taluka, Aravalli District, Gujarat, India.</p></div>""",
        unsafe_allow_html=True,
    )
    if not any(v is not None for v in ds.values()):
        st.error("No project files were found. Add your Module 01-05 files to the "
                 "`data/` and `outputs/` folders of the repository (see the Data Checks page).")
    kpis = compute_kpis()
    cols = st.columns(len(kpis))
    for col, (k, v) in zip(cols, kpis.items()):
        col.metric(k, v)
    st.markdown("### Modules")
    mods = [
        ("01", "AOI Verification", "Boundary preparation and quality checks"),
        ("02", "Satellite Indices", "Sentinel-2 NDVI, NDWI, MNDWI, NDMI"),
        ("03", "Rainfall & Terrain", "CHIRPS, SRTM elevation, slope, drainage"),
        ("04", "Stress & GeoAI", "Relative stress screening, KMeans clusters"),
        ("05", "Scenarios", "Priority scoring and buffer coverage"),
        ("06", "Dashboard", "Maps, analytics and report downloads"),
    ]
    for col, (n, t, d) in zip(st.columns(6), mods):
        col.markdown(f"**{n} · {t}**  \n<span style='color:#a8bdc7;font-size:13px'>{d}</span>",
                     unsafe_allow_html=True)
    st.markdown(
        "<div class='tv-note'>Prototype for screening and planning support. Not a live "
        "sensor-connected or physically calibrated digital twin. Stress classes are "
        "relative to the study area.</div>",
        unsafe_allow_html=True,
    )


def page_map():
    st.header("GIS Map")
    aoi = get_vector("AOI Boundary")
    avail = [n for n in RASTERS if ds.get(n) is not None]
    c1, c2, c3 = st.columns([3, 1, 1])
    chosen = c1.multiselect("Raster layers", avail, default=avail[:1])
    opacity = c2.slider("Opacity", 0.1, 1.0, 0.75, 0.05)
    tiles = c3.selectbox("Base map", ["OpenStreetMap", "Satellite (Esri)"])
    c4, c5 = st.columns(2)
    show_c = c4.checkbox("Intervention candidates", value=True)
    show_b = c5.checkbox("Saved scenario buffers", value=False)
    m = new_map(aoi, tiles)
    for n in chosen:
        add_raster(m, n, opacity)
    if show_c:
        cand = get_vector("Intervention Candidates")
        if cand is not None:
            add_points(m, cand)
    if show_b:
        buf = get_vector("Scenario Buffers")
        if buf is not None:
            folium.GeoJson(
                geom_only(buf, 0.0002).to_json(), name="Scenario buffers",
                style_function=lambda x: {"color": "#ff7f50", "weight": 1, "fillOpacity": 0.15},
            ).add_to(m)
    show_map(m)


def page_indices():
    st.header("Satellite Indices")
    st.caption("Statistics are computed on a down-sampled preview (max "
               f"{MAX_DIM}px) for speed.")
    tabs = st.tabs(INDEX_NAMES)
    rows = []
    for tab, name in zip(tabs, INDEX_NAMES):
        with tab:
            r = get_raster(name)
            if r is None:
                continue
            data, bounds = r
            st_ = stats_of(data)
            rows.append({"Index": name, **{k: round(v, 4) for k, v in st_.items()}})
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = raster_figure(name, data, bounds)
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                st.dataframe(pd.DataFrame(st_, index=["value"]).T.round(4))
                v = data[np.isfinite(data)]
                hist = px.histogram(x=v, nbins=40, labels={"x": name}, template="plotly_dark")
                hist.update_traces(marker_color="#35c6b5")
                st.plotly_chart(hist)
    if rows:
        st.subheader("Index comparison")
        st.dataframe(pd.DataFrame(rows).set_index("Index"))


def stress_classes(data):
    v = data[np.isfinite(data)]
    uniq = np.unique(v)
    if len(uniq) <= 6 and np.allclose(uniq, np.round(uniq)):
        labels = [f"Class {int(u)}" for u in uniq]
        counts = [(v == u).sum() for u in uniq]
    else:
        edges = np.quantile(v, [0, 0.25, 0.5, 0.75, 1.0])
        edges = np.unique(edges)
        if len(edges) < 3:
            return None
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
        names = ["Low", "Moderate", "High", "Very high"][: len(edges) - 1]
        labels = names
        counts = [(idx == i).sum() for i in range(len(names))]
    pct = 100 * np.array(counts) / np.sum(counts)
    return pd.DataFrame({"Class": labels, "Share of area (%)": pct.round(1)})


def page_stress():
    st.header("Agricultural Stress")
    r = get_raster("Agricultural Stress")
    if r is not None:
        data, bounds = r
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = raster_figure("Agricultural Stress", data, bounds)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            cls = stress_classes(data)
            if cls is not None:
                st.subheader("Relative stress classes")
                bar = px.bar(cls, x="Class", y="Share of area (%)", template="plotly_dark")
                bar.update_traces(marker_color="#e76f51")
                st.plotly_chart(bar)
                st.dataframe(cls.set_index("Class"))
            st.dataframe(pd.DataFrame(stats_of(data), index=["value"]).T.round(4))
        st.markdown(
            "<div class='tv-note'>Stress is a <b>relative</b> screening score built from "
            "satellite indicators. Quartile classes compare areas within Bayad only.</div>",
            unsafe_allow_html=True,
        )
    tbl = get_table("Stress Summary", quiet=True)
    if tbl is not None:
        st.subheader("Stress summary table (Module 04)")
        st.dataframe(tbl)


def page_clusters():
    st.header("GeoAI Clusters")
    r = get_raster("GeoAI Clusters")
    if r is None:
        return
    data, bounds = r
    vals, counts = np.unique(data[np.isfinite(data)].astype(int), return_counts=True)
    share = pd.DataFrame({"Cluster": vals, "Share of area (%)": (100 * counts / counts.sum()).round(1)})
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = raster_figure("GeoAI Clusters", data, bounds)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        pie = px.pie(share, names="Cluster", values="Share of area (%)", template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(pie)
        st.dataframe(share.set_index("Cluster"))
    # cluster profile (only when other rasters share the exact preview grid)
    prof = {}
    for nm in ["NDVI", "NDMI", "Agricultural Stress"]:
        o = get_raster(nm, quiet=True)
        if o is not None and o[0].shape == data.shape:
            prof[nm] = o[0]
    if prof:
        rows = []
        for v in vals:
            mask = (data == v)
            row = {"Cluster": v}
            for nm, arr in prof.items():
                row[f"Mean {nm}"] = round(float(np.nanmean(arr[mask])), 3)
            rows.append(row)
        st.subheader("Cluster profile")
        st.dataframe(pd.DataFrame(rows).set_index("Cluster"))
    tbl = get_table("Cluster Summary", quiet=True)
    if tbl is not None:
        st.subheader("Cluster summary table (Module 04)")
        st.dataframe(tbl)
    st.markdown(
        "<div class='tv-note'>KMeans clusters are unsupervised, exploratory groupings. "
        "Interpret them with field knowledge before drawing agronomic conclusions.</div>",
        unsafe_allow_html=True,
    )


def page_scenarios():
    st.header("Digital Twin Scenarios")
    tbl = get_table("Scenario Summary", quiet=True)
    if tbl is not None:
        st.subheader("Saved scenario results (Module 05)")
        st.dataframe(tbl)
        num = tbl.select_dtypes("number").columns.tolist()
        if num and len(tbl) > 1:
            label = tbl.columns[0]
            fig = px.bar(tbl, x=label, y=num[:3], barmode="group", template="plotly_dark")
            st.plotly_chart(fig)

    cand = get_vector("Intervention Candidates")
    aoi = get_vector("AOI Boundary")
    if cand is None or aoi is None:
        st.info("The interactive what-if needs both the AOI boundary and the intervention candidates.")
        return

    st.subheader("What-if: buffer coverage")
    pts = cand.to_crs(PROJECT_CRS)
    pts["geometry"] = pts.geometry.representative_point()
    aoi_p = aoi.to_crs(PROJECT_CRS)
    aoi_geom = union(aoi_p.geometry)
    score_col = next(
        (c for c in pts.columns if c != "geometry"
         and any(k in c.lower() for k in ("priority", "score"))
         and pd.api.types.is_numeric_dtype(pts[c])), None)

    c1, c2, c3 = st.columns(3)
    radius = c1.slider("Buffer radius (m)", 100, 3000, 500, 50)
    n_all = len(pts)
    top_n = c2.slider("Top candidates", 1, n_all, min(20, n_all)) if n_all > 1 else 1
    c3.metric("Ranking column", score_col or "first rows (no score column)")
    sel = pts.sort_values(score_col, ascending=False).head(top_n) if score_col else pts.head(top_n)

    def coverage(r):
        u = union(sel.geometry.buffer(r))
        return 100 * u.intersection(aoi_geom).area / aoi_geom.area

    m1, m2 = st.columns(2)
    m1.metric("AOI covered by buffers", f"{coverage(radius):.1f}%")
    m2.metric("Candidates used", str(len(sel)))

    sweep = pd.DataFrame({"Radius (m)": [250, 500, 1000, 1500, 2000, 3000]})
    sweep["Coverage (%)"] = [round(coverage(r), 2) for r in sweep["Radius (m)"]]
    fig = px.line(sweep, x="Radius (m)", y="Coverage (%)", markers=True, template="plotly_dark")
    fig.update_traces(line_color="#35c6b5")
    st.plotly_chart(fig)

    m = new_map(aoi)
    buf = gpd.GeoDataFrame(geometry=sel.geometry.buffer(radius), crs=PROJECT_CRS)
    folium.GeoJson(
        geom_only(buf, 0.0002).to_json(), name=f"Buffers {radius} m",
        style_function=lambda x: {"color": "#ff7f50", "weight": 1, "fillOpacity": 0.2},
    ).add_to(m)
    add_points(m, sel, "Selected candidates")
    show_map(m, 560)
    st.markdown(
        "<div class='tv-note'>Coverage is a geometric screening measure (buffer area inside "
        "the AOI). It does not model crop, hydrologic or economic outcomes.</div>",
        unsafe_allow_html=True,
    )


def page_downloads():
    st.header("Download Reports")
    kpis = compute_kpis()
    c1, c2 = st.columns(2)
    kdf = pd.DataFrame({"Indicator": list(kpis.keys()), "Value": list(kpis.values())})
    c1.download_button("Summary CSV", kdf.to_csv(index=False).encode(),
                       "terravita_bayad_summary.csv", "text/csv")
    pdf = build_pdf(kpis)
    if pdf:
        c2.download_button("Summary PDF", pdf, "terravita_bayad_summary.pdf", "application/pdf")
    else:
        c2.info("Add `reportlab` to requirements.txt to enable the PDF summary.")

    st.subheader("Source layers")
    shown = False
    for name, p in ds.items():
        if p is None:
            continue
        try:
            size = p.stat().st_size
            if size > 100 * 1024 * 1024:
                st.caption(f"{name}: file too large for download here ({size / 1e6:.0f} MB).")
                continue
            mime = {".tif": "image/tiff", ".tiff": "image/tiff", ".csv": "text/csv",
                    ".geojson": "application/geo+json"}.get(p.suffix.lower(), "application/octet-stream")
            st.download_button(f"{name} ({p.name})", p.read_bytes(), p.name, mime, key=f"dl_{name}")
            shown = True
        except Exception as e:
            st.warning(f"{name}: {e}")
    if not shown:
        st.info("No source files available to download.")


def page_checks():
    st.header("Data Checks")
    st.caption("Searching in: " + ", ".join(f"`{rel(d)}/`" for d in SEARCH_DIRS))
    rows = []
    for name in ds:
        p = ds[name]
        kind = "Raster" if name in RASTERS else ("Vector" if name in VECTORS else "Table")
        row = {"Layer": name, "Type": kind, "Status": "Missing", "File": "", "Details": ""}
        if p is not None:
            row["File"] = rel(p)
            row["Status"] = "Found"
            if kind == "Raster":
                v = validate_raster(str(p))
                row["Status"] = "OK" if v["ok"] else "Error"
                parts = []
                if v.get("crs"):
                    parts.append(v["crs"])
                if "width" in v:
                    parts.append(f"{v['width']}x{v['height']} px, {v['bands']} band(s)")
                if "valid_pct" in v:
                    parts.append(f"{v['valid_pct']}% valid")
                parts += v["errors"] + v["warnings"]
                row["Details"] = "; ".join(parts)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df.set_index("Layer"))
    miss = df[df["Status"] == "Missing"]["Layer"].tolist()
    if miss:
        st.warning("Missing: " + ", ".join(miss) +
                   ". Check file names (case-sensitive on GitHub) or set OVERRIDES in app.py.")


# ----------------------------------------------------------------------------
# NAVIGATION
# ----------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "GIS Map": page_map,
    "Satellite Indices": page_indices,
    "Agricultural Stress": page_stress,
    "GeoAI Clusters": page_clusters,
    "Digital Twin Scenarios": page_scenarios,
    "Download Reports": page_downloads,
    "Data Checks": page_checks,
}

with st.sidebar:
    st.markdown("## 🌍 TERRAVITA")
    st.caption("Environmental & Geospatial Consultancy")
    choice = st.radio("Navigate", list(PAGES.keys()))
    found = sum(v is not None for v in ds.values())
    st.markdown("---")
    st.caption(f"Data layers found: {found} of {len(ds)}")
    st.caption("Bayad Taluka, Aravalli, Gujarat")

try:
    PAGES[choice]()
except Exception as exc:  # last-resort guard so the app never shows a raw traceback
    st.error(f"Something went wrong on this page: {exc}")
    st.info("Open the Data Checks page to confirm all files and rasters are valid.")
