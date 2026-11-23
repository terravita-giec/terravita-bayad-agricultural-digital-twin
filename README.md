## Live Demo
- Landing page: https://YOUR-USERNAME.github.io/YOUR-REPO/
- Interactive dashboard: https://YOUR-APP.streamlit.app

## Run locally
pip install -r requirements.txt
streamlit run app.py

# TerraVita | AI-Driven Agricultural Digital Twin

### Watershed Resilience and Agricultural Stress Assessment — Bayad, Gujarat, India

An exploratory geospatial decision-support prototype combining
satellite remote sensing, GIS, terrain analysis and GeoAI to
investigate agricultural stress patterns and spatial priorities
for watershed resilience planning.

Developed by **TerraVita Environmental & Geospatial Consultancy**.

---

## 1. Project Overview

Agricultural landscapes are influenced by vegetation conditions,
rainfall variability, terrain characteristics and drainage patterns.

This project integrates satellite-derived indicators and
environmental datasets to develop a spatial screening framework
for agricultural stress assessment and preliminary intervention
planning.

## 2. Study Area

- Location: Bayad Taluka
- District: Aravalli
- State: Gujarat
- Country: India
- Pilot season: Kharif 2025

The administrative boundary is based on the post-2025
taluka reorganisation and subsequent village amendment.
The boundary composition and geometry should be verified
against the official administrative records.

## 3. Objectives

- Analyse seasonal vegetation conditions.
- Assess vegetation moisture-related indicators.
- Examine rainfall variability.
- Derive terrain and drainage characteristics.
- Identify relative agricultural stress patterns.
- Explore spatial clusters using unsupervised GeoAI.
- Develop preliminary intervention priority zones.
- Compare spatial buffer coverage scenarios.

## 4. Data Sources

| Dataset | Application |
|---|---|
| Sentinel-2 SR Harmonized | Vegetation and moisture indices |
| CHIRPS Daily | Rainfall analysis |
| SRTM DEM | Elevation and slope |
| Survey of India village boundaries | Administrative AOI |
| Derived GIS layers | Drainage and spatial analysis |

See `docs/data_sources.md` for dataset details,
versions, access links and processing notes.

## 5. Technical Workflow

1. Administrative boundary preparation and verification.
2. Sentinel-2 seasonal composites.
3. NDVI, NDWI, MNDWI and NDMI calculation.
4. Rainfall and anomaly analysis.
5. DEM processing and drainage derivation.
6. Relative agricultural stress screening.
7. Exploratory KMeans clustering.
8. Spatial priority assessment.
9. Intervention buffer scenario analysis.
10. Interactive dashboard development.

## 6. Technology Stack

- Google Earth Engine
- Python
- Google Colab
- Rasterio
- GeoPandas
- NumPy
- Pandas
- Scikit-learn
- WhiteboxTools
- Streamlit
- Plotly
- Folium
- QGIS

## 7. Dashboard

The Streamlit dashboard provides:

- Satellite indicator exploration
- Agricultural stress visualisation
- Rainfall and terrain analysis
- Drainage proximity mapping
- Preliminary intervention priorities
- Candidate field-investigation locations
- Buffer scenario comparison

## 8. Repository Structure

- `app/` — Interactive dashboard
- `notebooks/` — Processing workflows
- `scripts/` — Reusable Python scripts
- `docs/` — Methodology and data documentation
- `outputs/` — Selected maps and charts
- `assets/` — Project images and branding

## 9. Scientific Limitations

This project is an exploratory geospatial screening
and decision-support prototype.

- NDVI is not a direct crop-yield measurement.
- NDMI is not a direct groundwater measurement.
- CHIRPS rainfall has much coarser spatial resolution
  than Sentinel-2.
- DEM-derived drainage requires field verification.
- KMeans clusters are not validated land-cover classes.
- Priority weights are exploratory assumptions.
- Candidate points are not approved construction sites.
- Buffer coverage does not establish recharge or
  hydrological effectiveness.

Field observations, agronomic data, hydrogeological
investigation and independent validation are required
before operational decisions.

## 10. Developer

**TerraVita Environmental & Geospatial Consultancy**

Satellite + GIS + AI for Environmental, Urban
and Geological Decision-Making.

---

*Research and portfolio demonstration project.*
