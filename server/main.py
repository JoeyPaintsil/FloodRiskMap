# server/main.py
# Run locally with Docker: uvicorn server.main:app --host 0.0.0.0 --port 8000

import os
import math
import time
import json
import tempfile
from typing import Dict, Any, Optional
from urllib.parse import urljoin
from pathlib import Path

# Clean PROJ/GDAL env to avoid conflicts on some systems
for var in ("PROJ_LIB", "GDAL_DATA", "GDAL_DRIVER_PATH", "PROJ_NETWORK"):
    os.environ.pop(var, None)
try:
    import pyproj  # optional
    os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
except Exception:
    pass

import ee
import numpy as np
import pandas as pd
import requests
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent            # .../server
ROOT_DIR = BASE_DIR.parent                            # project root

# REQUIRED CREDENTIALS (set these in your .env / Render env):
#   EE_PROJECT            = 103526698840
#   EE_SA_EMAIL           = ...@....iam.gserviceaccount.com
#   EE_SA_KEY_JSON_B64    = <base64 of the FULL JSON key>   (preferred)
#   or EE_SA_KEY_JSON     = <FULL JSON key on ONE line>     (fallback)
EE_PROJECT = os.getenv("EE_PROJECT")
EE_SA_EMAIL = os.getenv("EE_SA_EMAIL")
EE_SA_KEY_JSON = os.getenv("EE_SA_KEY_JSON")
EE_SA_KEY_JSON_B64 = os.getenv("EE_SA_KEY_JSON_B64")

# ======== IMPORTANT: point this to your training CSV ========
TRAIN_CSV = os.getenv(
    "TRAIN_CSV",
    str(BASE_DIR / "data" / "Geomundus_Training_Points_v1.csv")  # <--- change if you prefer hardcoded path
)
# ============================================================

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_DEFAULT = 2018
DIMENSIONS_DEFAULT = 2048  # PNG render size for dashboard

# Features used for BOTH training and prediction (must match CSV columns)
FEATURE_COLS = [
    "rainfall_mm", "ndvi", "lulc_igbp", "elevation", "slope",
    "curvature_profile", "twi", "drainage_density", "dist_to_water"
]
LABEL_COL = "label"

# Risk palette (exact colors you provided; used for PNG->GeoTIFF decoding)
RISK_COLORS_HEX = ["006400", "7FFF00", "FFFF00", "FFA500", "FF0000"]  # 0..4
RISK_COLORS_RGB = np.array(
    [[int(h[i:i+2], 16) for i in (0, 2, 4)] for h in RISK_COLORS_HEX],
    dtype=np.uint8
)

# ---------- FastAPI ----------
app = FastAPI(title="Flood Mapper API (PNG→GeoTIFF via color codes)", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend domains in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

@app.get("/")
def root():
    return {"ok": True, "msg": "Use POST /predict and POST /resolve_area"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/files/{filename}")
def get_file(filename: str):
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)

# ---------- EE init with Service Account ----------
def init_ee_with_service_account():
    if not EE_PROJECT or not EE_SA_EMAIL or not (EE_SA_KEY_JSON_B64 or EE_SA_KEY_JSON):
        raise RuntimeError("Missing EE_PROJECT, EE_SA_EMAIL, or SA key (B64 or raw JSON).")

    if EE_SA_KEY_JSON_B64:
        import base64
        key_json_text = base64.b64decode(EE_SA_KEY_JSON_B64).decode("utf-8")
    else:
        key_json_text = EE_SA_KEY_JSON

    if key_json_text.strip().startswith("-----BEGIN"):
        raise RuntimeError("EE key must be the FULL service-account JSON, not just the private_key block.")

    try:
        data = json.loads(key_json_text)
        if not isinstance(data, dict) or "client_email" not in data or "private_key" not in data:
            raise ValueError("Not a valid service account JSON.")
    except Exception as e:
        raise RuntimeError(f"Service account JSON is invalid: {e}")

    key_path = os.path.join(tempfile.gettempdir(), "ee-key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(key_json_text)

    creds = ee.ServiceAccountCredentials(EE_SA_EMAIL, key_path)
    ee.Initialize(credentials=creds, project=EE_PROJECT)

try:
    init_ee_with_service_account()
except Exception as e:
    raise RuntimeError(f"Earth Engine init failed: {e!r}")

# ---------- Helpers ----------
def _row_to_feature(row: Dict[str, Any]) -> ee.Feature:
    """CSV row -> ee.Feature with numeric properties matching FEATURE_COLS + label."""
    return ee.Feature(None, {
        "rainfall_mm": float(row["rainfall_mm"]),
        "ndvi": float(row["ndvi"]),
        "lulc_igbp": int(row["lulc_igbp"]),
        "elevation": float(row["elevation"]),
        "slope": float(row["slope"]),
        "curvature_profile": float(row["curvature_profile"]),
        "twi": float(row["twi"]),
        "drainage_density": float(row["drainage_density"]),
        "dist_to_water": float(row["dist_to_water"]),
        "label": int(row["label"]),
    })

jrc_occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

def rainfall_sum_year(year: int):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).sum().rename("rainfall_mm")

def ndvi_s2_mean_year(year: int, region: ee.Geometry):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    def mask_s2_clouds(im): return im.updateMask(im.select("QA60").lt(1))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(start, end).filterBounds(region)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
          .map(mask_s2_clouds))
    return s2.map(lambda im: im.normalizedDifference(["B8","B4"]).rename("NDVI")).select("NDVI").mean().rename("ndvi")

def ndvi_modis_mean_year(year: int, region: ee.Geometry):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    return ee.ImageCollection("MODIS/006/MOD13Q1").filterDate(start, end).filterBounds(region).select("NDVI").mean().multiply(0.0001).rename("ndvi")

def ndvi_mean_year(year: int, region: ee.Geometry):
    # MODIS before 2015, S2 after
    return ee.Image(ee.Algorithms.If(ee.Number(year).lt(2015),
                                     ndvi_modis_mean_year(year, region),
                                     ndvi_s2_mean_year(year, region)))

def modis_lulc_year(year: int, region: ee.Geometry):
    yc = ee.Number(year).max(2001).min(2020)
    start = ee.Date.fromYMD(yc, 1, 1); end = ee.Date.fromYMD(yc, 12, 31)
    img = ee.ImageCollection("MODIS/006/MCD12Q1").filterDate(start, end).filterBounds(region).select("LC_Type1").first()
    return ee.Image(img).rename("lulc_igbp")

def profile_curvature_from_dem(dem: ee.Image):
    proj = dem.projection(); scale = ee.Number(proj.nominalScale())
    dx = dem.convolve(ee.Kernel.fixed(3, 3, [[-1,0,1],[-2,0,2],[-1,0,1]], 1, 1, False)).divide(ee.Number(8).multiply(scale))
    dy = dem.convolve(ee.Kernel.fixed(3, 3, [[-1,-2,-1],[0,0,0],[1,2,1]], 1, 1, False)).divide(ee.Number(8).multiply(scale))
    dxx = dem.convolve(ee.Kernel.fixed(3, 3, [[1,-2,1],[2,-4,2],[1,-2,1]], 1, 1, False)).divide(ee.Number(6).multiply(scale.pow(2)))
    dyy = dem.convolve(ee.Kernel.fixed(3, 3, [[1,2,1],[-2,-4,-2],[1,2,1]], 1, 1, False)).divide(ee.Number(6).multiply(scale.pow(2)))
    dxy = dem.convolve(ee.Kernel.fixed(3, 3, [[1,0,-1],[0,0,0],[-1,0,1]], 1, 1, False)).divide(ee.Number(4).multiply(scale.pow(2)))
    p, q, r, t, s = dx, dy, dxx, dyy, dxy
    denom = p.multiply(p).add(q.multiply(q)).add(1).pow(1.5)
    return (r.multiply(p.pow(2)).add(t.multiply(q.pow(2))).add(s.multiply(2).multiply(p).multiply(q))).divide(denom).rename("curvature_profile")

def build_static_stack(region: ee.Geometry):
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").clip(region).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    curv = profile_curvature_from_dem(dem)
    upa = ee.Image("MERIT/Hydro/v1_0_1").select("upa").clip(region)
    tan_slope = slope.multiply(math.pi / 180).tan().max(0.001)
    cell_area = 90 * 90
    twi = upa.multiply(cell_area).divide(tan_slope).add(1).log().rename("twi")
    stream_threshold = 1000
    streams = upa.gt(stream_threshold)
    pixel_size = 90
    kernel_radius_m = 1000
    kernel_radius_px = int(kernel_radius_m / pixel_size)
    kernel = ee.Kernel.circle(kernel_radius_px, units="pixels", normalize=False)
    stream_density_pixels = streams.convolve(kernel)
    dd = stream_density_pixels.multiply(pixel_size).divide((kernel_radius_m ** 2) / 1e6).rename("drainage_density")
    water_perm = jrc_occurrence.clip(region).gt(90)
    dist_to_water = water_perm.Not().fastDistanceTransform(30).sqrt().multiply(30).rename("dist_to_water")
    return ee.Image.cat([dem, slope, curv, twi, dd, dist_to_water]).clip(region)

def predictor_image(region: ee.Geometry, year: int):
    rain_img = rainfall_sum_year(year).clip(region)
    ndvi_img = ndvi_mean_year(year, region).clip(region)
    lulc_img = modis_lulc_year(year, region).clip(region).toInt()
    static_stack = build_static_stack(region)
    predictor_img = ee.Image.cat([rain_img, ndvi_img, lulc_img, static_stack]).select(FEATURE_COLS)
    dem_proj = ee.Image("NASA/NASADEM_HGT/001").select("elevation").projection()
    continuous = predictor_img.select([
        "rainfall_mm","ndvi","elevation","slope",
        "curvature_profile","twi","drainage_density","dist_to_water"
    ]).resample("bilinear").reproject(dem_proj)
    lulc_nearest = predictor_img.select("lulc_igbp").reproject(dem_proj)
    return ee.Image.cat([continuous, lulc_nearest]).select(FEATURE_COLS)

# ---------- Train RF in EE from CSV ----------
def train_classifier_from_csv():
    if not os.path.exists(TRAIN_CSV):
        raise RuntimeError(f"Training CSV not found at TRAIN_CSV={TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)

    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {missing}")

    df_use = df[FEATURE_COLS + [LABEL_COL]].dropna().copy()
    feats = [_row_to_feature(r) for r in df_use.to_dict("records")]
    fc = ee.FeatureCollection(feats).randomColumn('rand', 42)
    train_fc = fc.filter(ee.Filter.lt('rand', 0.8))

    clf = (ee.Classifier.smileRandomForest(
                numberOfTrees=300,
                minLeafPopulation=2,
                seed=42
           )
           .train(features=train_fc, classProperty=LABEL_COL, inputProperties=FEATURE_COLS))
    return clf, clf.setOutputMode('PROBABILITY')

CLASSIFIER, CLASSIFIER_PROB = train_classifier_from_csv()

# ---------- Schemas ----------
class PredictBody(BaseModel):
    year: Optional[int] = YEAR_DEFAULT
    dimensions: Optional[int] = DIMENSIONS_DEFAULT
    aoi_coords: Optional[list] = None        # [[lon,lat], ...] outer ring
    named_area: Optional[dict] = None        # { level: "region"|"district", name: str, country: str }
    mode: Optional[str] = "prob"             # "prob" (dashboard) or "risk" (dashboard)

class PredictOut(BaseModel):
    png_url: str
    tif_url: str
    bounds: Dict[str, float]   # {minx,miny,maxx,maxy}
    width: int
    height: int
    crs: str
    message: str

class ResolveAreaBody(BaseModel):
    named_area: dict  # { level, name, country }

class ResolveAreaOut(BaseModel):
    bounds: Dict[str, float]
    geojson: dict

# ---------- GAUL resolvers ----------
def _load_gaul(level: int) -> ee.FeatureCollection:
    sources = [
        f"FAO/GAUL/2015/level{level}",
        f"FAO/GAUL_SIMPLIFIED_500m/2015/level{level}",
    ]
    last_err = None
    for src in sources:
        try:
            return ee.FeatureCollection(src)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"GAUL level{level} dataset not available: {last_err}")

def resolve_named_area_geom(named: dict) -> ee.Geometry:
    if not named or "level" not in named or "name" not in named or "country" not in named:
        raise HTTPException(400, "Invalid named_area payload. Expect {level,name,country}.")
    level = str(named["level"]).lower().strip()
    name = str(named["name"]).strip()
    country = str(named["country"]).strip()
    if level == "region":
        fc = _load_gaul(1)
        feats = (fc.filter(ee.Filter.eq("ADM0_NAME", country))
                   .filter(ee.Filter.eq("ADM1_NAME", name)))
    elif level == "district":
        fc = _load_gaul(2)
        feats = (fc.filter(ee.Filter.eq("ADM0_NAME", country))
                   .filter(ee.Filter.eq("ADM2_NAME", name)))
    else:
        raise HTTPException(400, "named_area.level must be 'region' or 'district'.")
    if ee.Number(feats.size()).getInfo() == 0:
        raise HTTPException(404, f"GAUL feature not found for {level} '{name}' in {country}.")
    return feats.geometry()

# ---------- PNG fetch ----------
def _fetch_png(visual_img, region, dims):
    params = {
        "region": region,
        "dimensions": int(dims),
        "format": "png",
        "backgroundColor": "00000000",
    }
    url = visual_img.getThumbURL(params)
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"EE PNG download failed: {r.status_code} {r.text[:200]}")
    return r.content

# ---------- Color-based PNG → GeoTIFF (risk classes 0..4) ----------
def _risk_png_to_geotiff(png_path: str, out_tif_path: str, bounds: Dict[str, float]):
    """Decode risk classes from risk PNG colors and write a class GeoTIFF (0..4, 255 nodata)."""
    with Image.open(png_path) as im:
        im = im.convert("RGBA")
        rgba = np.array(im)
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3].astype(np.int16)  # avoid uint8 overflow during diff

    # Vectorized nearest-color classification
    # Shape: (H, W, 3) -> (5, H, W) distances to palette colors
    diffs = rgb[None, :, :, :] - RISK_COLORS_RGB[:, None, None, :].astype(np.int16)
    d2 = np.sum(diffs * diffs, axis=3)          # squared distance
    cls = np.argmin(d2, axis=0).astype(np.uint8)  # 0..4

    # Set nodata where alpha == 0
    cls[alpha == 0] = 255

    transform = from_bounds(bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"], width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": width,
        "height": height,
        "transform": transform,
        "crs": CRS.from_epsg(4326),
        "nodata": 255,
        "compress": "lzw"
    }
    with rasterio.open(out_tif_path, "w", **profile) as dst:
        dst.write(cls, 1)
    return width, height

# ---------- Helper for client-side AOI preview ----------
@app.post("/resolve_area", response_model=ResolveAreaOut)
def resolve_area(req: ResolveAreaBody):
    try:
        geom = resolve_named_area_geom(req.named_area)
        bcoords = geom.bounds().coordinates().getInfo()[0]
        lons = [c[0] for c in bcoords]; lats = [c[1] for c in bcoords]
        minx, maxx = min(lons), max(lons); miny, maxy = min(lats), max(lats)
        gjson = geom.getInfo()
        return ResolveAreaOut(bounds={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}, geojson=gjson)
    except HTTPException:
        raise
    except Exception as e:
        print("resolve_area error:", repr(e))
        raise HTTPException(500, str(e))

# ---------- Main endpoint ----------
@app.post("/predict", response_model=PredictOut)
def predict(req: PredictBody, request: Request):
    try:
        year = int(req.year or YEAR_DEFAULT)
        dims = int(req.dimensions or DIMENSIONS_DEFAULT)
        mode = (req.mode or "prob").lower().strip()  # "prob" or "risk" for the dashboard PNG

        if req.aoi_coords:
            uploaded_aoi = ee.Geometry.Polygon([req.aoi_coords])
            region_geom = uploaded_aoi
        elif req.named_area:
            region_geom = resolve_named_area_geom(req.named_area)
        else:
            raise HTTPException(400, "Provide either aoi_coords or named_area.")

        # Build predictors, classify on the server
        pred_img = predictor_image(region_geom, year)

        # Probability (0..1) and 5-class risk (0..4)
        prob_img = pred_img.classify(CLASSIFIER_PROB).rename("flood_prob")
        risk_img = (ee.Image(0)
                    .where(prob_img.gte(0.2), 1)
                    .where(prob_img.gte(0.4), 2)
                    .where(prob_img.gte(0.6), 3)
                    .where(prob_img.gte(0.8), 4)
                    .rename("flood_risk"))

        # Visualization images
        prob_palette = ["006400","7FFF00","FFFF00","FFA500","FF0000"]
        vis_prob = prob_img.visualize(min=0, max=1, palette=prob_palette)
        vis_risk = risk_img.visualize(min=0, max=4, palette=prob_palette)

        ts = int(time.time())
        dash_suffix = "prob" if mode == "prob" else "risk"
        png_dash = f"flood_{dash_suffix}_{ts}.png"     # dashboard PNG (what user sees)
        png_risk = f"flood_risk_{ts}.png"              # risk PNG used for GeoTIFF decoding
        tif_name = f"flood_risk_{ts}.tif"              # final GeoTIFF (classes 0..4)
        png_dash_path = os.path.join(OUTPUT_DIR, png_dash)
        png_risk_path = os.path.join(OUTPUT_DIR, png_risk)
        tif_path = os.path.join(OUTPUT_DIR, tif_name)

        # Download dashboard PNG
        dash_img = vis_prob if mode == "prob" else vis_risk
        png_bytes = _fetch_png(dash_img, region_geom, dims)
        with open(png_dash_path, "wb") as f:
            f.write(png_bytes)

        # Also download the discrete risk PNG (always used for GeoTIFF conversion)
        risk_png_bytes = _fetch_png(vis_risk, region_geom, dims)
        with open(png_risk_path, "wb") as f:
            f.write(risk_png_bytes)

        # AOI bounds for georeferencing
        bounds_geom = ee.Geometry(region_geom).bounds()
        coords = bounds_geom.coordinates().getInfo()[0]
        lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
        bounds = {"minx": min(lons), "miny": min(lats), "maxx": max(lons), "maxy": max(lats)}

        # Convert risk PNG to class GeoTIFF (0..4, 255 nodata) via color codes
        width, height = _risk_png_to_geotiff(png_risk_path, tif_path, bounds)

        base = str(request.base_url)  # e.g. http://localhost:8000/
        return PredictOut(
            png_url=urljoin(base, f"files/{png_dash}"),  # probability or risk, per 'mode'
            tif_url=urljoin(base, f"files/{tif_name}"),  # always risk classes GeoTIFF
            bounds=bounds,
            width=int(width),
            height=int(height),
            crs="EPSG:4326",
            message=f"Prediction complete (dashboard={dash_suffix}, GeoTIFF=risk classes 0..4)."
        )

    except HTTPException:
        raise
    except Exception as e:
        print("SERVER EXCEPTION:", repr(e))
        raise HTTPException(500, str(e))
