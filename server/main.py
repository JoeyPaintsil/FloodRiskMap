# server/main.py
#(in poweshell)
# # Activate (again, just in case)
#step1:  .\.venv\Scripts\Activate.ps1

# Verify which Python you’re about to use
# step2: python -c "import sys; print(sys.executable)"

# step 3: .\.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000

# Run locally: uvicorn server.main:app --host 0.0.0.0 --port 8000

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
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Load .env
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
print(f"🔍 Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)

# Debug print envs
print("EE_PROJECT:", os.getenv("EE_PROJECT"))
print("EE_SA_EMAIL:", os.getenv("EE_SA_EMAIL"))
print("EE_SA_KEY_JSON_B64 present?:", bool(os.getenv("EE_SA_KEY_JSON_B64")))

# EE credentials
EE_PROJECT = os.getenv("EE_PROJECT")
EE_SA_EMAIL = os.getenv("EE_SA_EMAIL")
EE_SA_KEY_JSON = os.getenv("EE_SA_KEY_JSON")
EE_SA_KEY_JSON_B64 = os.getenv("EE_SA_KEY_JSON_B64")

# Training CSV (now your new CSV)
TRAIN_CSV = os.getenv(
    "TRAIN_CSV",
    str(BASE_DIR / "data" / "Geomundus_Training_Points_v1.csv")
)

# Outputs
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_DEFAULT = 2018
DIMENSIONS_DEFAULT = 1024  # PNG render size

# --------- Feature set (matches your Colab) ----------
# Base numeric features
BASE_FEATURES = [
    "ndvi", "rainfall_mm",
    "dist_to_water", "elevation", "slope",
    "curvature_profile", "twi", "drainage_density",
]
# DynamicWorld one-hot (0..8)
LULC_CLASS_IDS = list(range(9))
LULC_DUMMIES = [f"lulc_{i}" for i in LULC_CLASS_IDS]

# Final features used for training and prediction
FEATURE_COLS = BASE_FEATURES + LULC_DUMMIES
LABEL_COL = "label"

# ---------- FastAPI ----------
app = FastAPI(title="Flood Mapper API (GAUL support)", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

@app.get("/")
def root():
    return {"ok": True, "msg": "See POST /predict, POST /resolve_area"}

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
        raise RuntimeError("Missing EE_PROJECT, EE_SA_EMAIL, or key env (EE_SA_KEY_JSON_B64 or EE_SA_KEY_JSON).")

    if EE_SA_KEY_JSON_B64:
        import base64
        try:
            key_json_text = base64.b64decode(EE_SA_KEY_JSON_B64).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Could not base64-decode EE_SA_KEY_JSON_B64: {e}")
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
    """Rows from CSV (with lulc_dw, not one-hot)."""
    return ee.Feature(None, {
        "ndvi": float(row["ndvi"]),
        "rainfall_mm": float(row["rainfall_mm"]),
        "dist_to_water": float(row["dist_to_water"]),
        "elevation": float(row["elevation"]),
        "slope": float(row["slope"]),
        "curvature_profile": float(row["curvature_profile"]),
        "twi": float(row["twi"]),
        "drainage_density": float(row["drainage_density"]),
        "lulc_dw": int(row["lulc_dw"]),
        "label": int(row["label"]),
    })

jrc_occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")

def rainfall_sum_year(year: int):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).sum().rename("rainfall_mm")

def ndvi_s2_mean_year(year: int, region: ee.Geometry):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    def mask_s2_clouds(im):
        scl = im.select('SCL')
        # Keep non-cloud classes
        mask = (scl.neq(3)  # cloud shadow
                .And(scl.neq(8))  # medium prob cloud
                .And(scl.neq(9))  # high prob cloud
                .And(scl.neq(10)) # thin cirrus
                .And(scl.neq(11)))# snow/ice
        return im.updateMask(mask)
    def add_ndvi(im): return im.addBands(im.normalizedDifference(["B8","B4"]).rename("NDVI"))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(start, end).filterBounds(region)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
          .map(mask_s2_clouds)
          .map(add_ndvi))
    return s2.select("NDVI").median().rename("ndvi")

def profile_curvature_from_dem(dem: ee.Image):
    proj = dem.projection(); scale = ee.Number(proj.nominalScale())
    dx  = dem.convolve(ee.Kernel.fixed(3,3,[[-1,0,1],[-2,0,2],[-1,0,1]],1,1,False)).divide(ee.Number(8).multiply(scale))
    dy  = dem.convolve(ee.Kernel.fixed(3,3,[[-1,-2,-1],[0,0,0],[1,2,1]],1,1,False)).divide(ee.Number(8).multiply(scale))
    dxx = dem.convolve(ee.Kernel.fixed(3,3,[[1,-2,1],[2,-4,2],[1,-2,1]],1,1,False)).divide(ee.Number(6).multiply(scale.pow(2)))
    dyy = dem.convolve(ee.Kernel.fixed(3,3,[[1,2,1],[-2,-4,-2],[1,2,1]],1,1,False)).divide(ee.Number(6).multiply(scale.pow(2)))
    dxy = dem.convolve(ee.Kernel.fixed(3,3,[[1,0,-1],[0,0,0],[-1,0,1]],1,1,False)).divide(ee.Number(4).multiply(scale.pow(2)))
    p, q, r, t, s = dx, dy, dxx, dyy, dxy
    denom = p.multiply(p).add(q.multiply(q)).add(1).pow(1.5)
    return (r.multiply(p.pow(2)).add(t.multiply(q.pow(2))).add(s.multiply(2).multiply(p).multiply(q))).divide(denom).rename("curvature_profile")

def build_static_stack(region: ee.Geometry):
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").clip(region).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    curv = profile_curvature_from_dem(dem)
    upa = ee.Image("MERIT/Hydro/v1_0_1").select("upa").clip(region)  # km^2
    tan_slope = slope.multiply(math.pi / 180).tan().max(0.001)
    cell_area = 90 * 90  # m^2
    twi = upa.multiply(cell_area).divide(tan_slope).add(1).log().rename("twi")
    # Streams & drainage density
    stream_threshold_km2 = 7  # similar to your Colab
    streams = upa.gte(stream_threshold_km2)
    pixel_size = 90
    kernel_radius_m = 1000
    kernel = ee.Kernel.circle(kernel_radius_m, units="meters", normalize=False)
    # Approximate drainage density (km / km^2)
    streams_m = streams.reproject(crs="EPSG:3857", scale=pixel_size)
    count_center_px = streams_m.unmask(0).reduceNeighborhood(ee.Reducer.sum(), kernel)
    length_km = count_center_px.multiply(pixel_size).divide(1000.0)
    area_km2 = ee.Number(math.pi * (kernel_radius_m ** 2) / 1e6)
    dd = length_km.divide(area_km2).rename("drainage_density")
    # Distance to permanent water (JRC occurrence >90)
    water_perm = jrc_occurrence.clip(region).gt(90).selfMask()
    dist_pix = water_perm.reproject(crs="EPSG:3857", scale=30).fastDistanceTransform(256).sqrt()
    pixel_m = ee.Number(30)
    dist_to_water = dist_pix.multiply(pixel_m).rename("dist_to_water")
    return ee.Image.cat([dem, slope, curv, twi, dd, dist_to_water]).clip(region)

def dynamicworld_lulc_onehot(year: int, region: ee.Geometry):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year, 12, 31)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(region).filterDate(start, end).select("label"))
    mode = dw.mode()  # 0..8
    dummies = [mode.eq(i).rename(f"lulc_{i}") for i in LULC_CLASS_IDS]
    return ee.Image.cat(dummies).clip(region)

def predictor_image(region: ee.Geometry, year: int):
    rain_img = rainfall_sum_year(year).clip(region)
    ndvi_img = ndvi_s2_mean_year(year, region).clip(region)
    static_stack = build_static_stack(region)  # elev, slope, curvature_profile, twi, drainage_density, dist_to_water
    dw_onehot = dynamicworld_lulc_onehot(year, region)   # lulc_0..lulc_8

    # Keep band order in FEATURE_COLS
    img = ee.Image.cat([
        ndvi_img.rename("ndvi"),
        rain_img.rename("rainfall_mm"),
        static_stack.select(["dist_to_water","elevation","slope","curvature_profile","twi","drainage_density"]),
        dw_onehot.select(LULC_DUMMIES),
    ])

    # Reproject to DEM projection for consistent scale
    dem_proj = ee.Image("NASA/NASADEM_HGT/001").select("elevation").projection()
    return img.reproject(dem_proj).select(FEATURE_COLS)

# ---------- Train RF ONCE (from new CSV) ----------
def train_classifier():
    if not os.path.exists(TRAIN_CSV):
        raise RuntimeError(f"Training CSV not found at {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)

    # Ensure required columns exist
    required = BASE_FEATURES + ["lulc_dw", LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {missing}")

    df_use = df[required].dropna().copy()
    df_use["lulc_dw"] = df_use["lulc_dw"].astype(int)
    df_use[LABEL_COL] = df_use[LABEL_COL].astype(int)

    feats = [_row_to_feature(r) for r in df_use.to_dict("records")]
    fc = ee.FeatureCollection(feats)

    # One-hot encode lulc_dw → lulc_0..lulc_8 on the server
    def add_lulc_dummies(feat):
        lulc = ee.Number(feat.get("lulc_dw")).toInt()
        d = ee.Dictionary.fromLists(LULC_DUMMIES, [lulc.eq(i) for i in LULC_CLASS_IDS])
        return feat.set(d)

    fc_encoded = fc.map(add_lulc_dummies)

    # 80/20 split (server-side)
    split = fc_encoded.randomColumn("rand", 42)
    train_fc = split.filter(ee.Filter.lt("rand", 0.8))

    clf = (ee.Classifier.smileRandomForest(numberOfTrees=300, minLeafPopulation=2, seed=42)
           .train(features=train_fc, classProperty=LABEL_COL, inputProperties=FEATURE_COLS))

    return clf

CLASSIFIER = train_classifier()

# ---------- Schemas ----------
class PredictBody(BaseModel):
    year: Optional[int] = YEAR_DEFAULT
    dimensions: Optional[int] = DIMENSIONS_DEFAULT
    aoi_coords: Optional[list] = None        # [[lon,lat], ...] outer ring
    named_area: Optional[dict] = None        # { level: "region"|"district", name: str, country: str }

class PredictOut(BaseModel):
    png_url: str
    tif_url: str
    bounds: Dict[str, float]
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

# ---------- PNG fetch with retry/downscale ----------
def _fetch_png_with_retries(visual_img, region, dims, max_retries=3):
    attempt = 0
    while True:
        params = {
            "region": region,
            "dimensions": int(dims),
            "format": "png",
            "backgroundColor": "00000000",
        }
        url = visual_img.getThumbURL(params)
        r = requests.get(url, timeout=180)
        if r.status_code == 200:
            return r.content, int(dims)
        txt = r.text or ""
        if r.status_code == 400 and "User memory limit" in txt and attempt < max_retries:
            dims = max(128, int(dims) // 2)
            attempt += 1
            continue
        raise RuntimeError(f"EE PNG download failed: {r.status_code} {txt[:200]}")

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

        if req.aoi_coords:
            uploaded_aoi = ee.Geometry.Polygon([req.aoi_coords])
            region_geom = uploaded_aoi
        elif req.named_area:
            region_geom = resolve_named_area_geom(req.named_area)
        else:
            raise HTTPException(400, "Provide either aoi_coords or named_area.")

        pred_img = predictor_image(region_geom, year)

        # --- PROBABILITY → 5 risk bins (unchanged display) ---
        prob_clf = CLASSIFIER.setOutputMode('PROBABILITY')
        prob_all = pred_img.select(FEATURE_COLS).classify(prob_clf)
        bn = prob_all.bandNames()

        prob = ee.Image(ee.Algorithms.If(
            bn.contains('probability_1'), prob_all.select('probability_1'),
            ee.Image(ee.Algorithms.If(
                bn.contains('probability'), prob_all.select('probability'),
                prob_all.select('classification').toFloat()
            ))
        )).rename('prob')

        idx = prob.multiply(5).floor().int16().max(0).min(4).rename('risk_idx')

        palette = ['22c55e', '86efac', 'fde047', 'f59e0b', 'ef4444']
        aoi_mask = ee.Image.constant(1).clip(region_geom)
        vis_img = idx.updateMask(aoi_mask).visualize(min=0, max=4, palette=palette)

        ts = int(time.time())
        png_name = f"flood_pred_{ts}.png"
        tif_name = f"flood_pred_{ts}.tif"
        png_path = os.path.join(OUTPUT_DIR, png_name)
        tif_path = os.path.join(OUTPUT_DIR, tif_name)

        png_bytes, used_dims = _fetch_png_with_retries(vis_img, region_geom, dims, max_retries=3)
        with open(png_path, "wb") as f:
            f.write(png_bytes)

        bounds = ee.Geometry(region_geom).bounds()
        coords = bounds.coordinates().getInfo()[0]
        lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
        minx, maxx = min(lons), max(lons); miny, maxy = min(lats), max(lats)

        with Image.open(png_path) as im:
            im = im.convert("RGBA")
            width, height = im.size
            rgba = np.array(im)

        # Map PNG colors → class indices 0..4; transparent → 255
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        palette_rgb = np.array([
            [0x22, 0xc5, 0x5e],  # 0
            [0x86, 0xef, 0xac],  # 1
            [0xfd, 0xe0, 0x47],  # 2
            [0xf5, 0x9e, 0x0b],  # 3
            [0xef, 0x44, 0x44],  # 4
        ], dtype=np.uint8)

        out_idx = np.full((height, width), 255, dtype=np.uint8)
        for k, color in enumerate(palette_rgb):
            match = (rgb[:, :, 0] == color[0]) & (rgb[:, :, 1] == color[1]) & (rgb[:, :, 2] == color[2])
            out_idx[match] = k
        out_idx[alpha == 0] = 255

        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": width,
            "height": height,
            "transform": transform,
            "crs": CRS.from_epsg(4326),
            "nodata": 255,
        }
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(out_idx, 1)

        base = str(request.base_url)
        return PredictOut(
            png_url=urljoin(base, f"files/{png_name}"),
            tif_url=urljoin(base, f"files/{tif_name}"),
            bounds={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
            width=width, height=height, crs="EPSG:4326",
            message=f"Prediction complete (dims used: {used_dims})."
        )

    except HTTPException:
        raise
    except Exception as e:
        print("SERVER EXCEPTION:", repr(e))
        raise HTTPException(500, str(e))
