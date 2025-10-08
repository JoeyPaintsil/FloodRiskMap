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
#   EE_PROJECT            = 103526698840                 # <-- your EE project id/number (PUT YOURS)
#   EE_SA_EMAIL           = ...@....iam.gserviceaccount.com   # <-- your service account email (PUT YOURS)
#   EE_SA_KEY_JSON_B64    = <base64 of the FULL JSON key>     # <-- preferred (PUT YOURS)
#   or EE_SA_KEY_JSON     = <FULL JSON key on ONE line>       # <-- optional fallback
EE_PROJECT = os.getenv("EE_PROJECT")
EE_SA_EMAIL = os.getenv("EE_SA_EMAIL")
EE_SA_KEY_JSON = os.getenv("EE_SA_KEY_JSON")              # optional: raw full JSON as a single line
EE_SA_KEY_JSON_B64 = os.getenv("EE_SA_KEY_JSON_B64")      # preferred: base64 of the full JSON

# Where to read training CSV from
TRAIN_CSV = os.getenv(
    "TRAIN_CSV",
    str(BASE_DIR / "data" / "gfd_samples_single_event_per_point.csv")
)

# Where to store outputs
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEAR_DEFAULT = 2018
DIMENSIONS_DEFAULT = 1024  # PNG render size

FEATURE_COLS = [
    "rainfall_mm", "ndvi", "lulc_igbp", "elevation", "slope",
    "curvature_profile", "twi", "drainage_density", "dist_to_water"
]
LABEL_COL = "label"

# ---------- FastAPI ----------
app = FastAPI(title="Flood Mapper API (GAUL support)", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your Vercel domain(s) for stricter security in production
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

    # Prefer base64 (best for .env files). Fallback to raw JSON string.
    if EE_SA_KEY_JSON_B64:
        import base64
        try:
            key_json_text = base64.b64decode(EE_SA_KEY_JSON_B64).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Could not base64-decode EE_SA_KEY_JSON_B64: {e}")
    else:
        key_json_text = EE_SA_KEY_JSON

    # Guard against pasting only the private_key block by mistake
    if key_json_text.strip().startswith("-----BEGIN"):
        raise RuntimeError("EE key must be the FULL service-account JSON, not just the private_key block.")

    # Validate JSON and required fields
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
    # Surface the error early so you know to set env vars correctly
    raise RuntimeError(f"Earth Engine init failed: {e!r}")

# ---------- Helpers ----------
def _row_to_feature(row: Dict[str, Any]) -> ee.Feature:
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

# ---------- Train RF ONCE ----------
def train_classifier():
    if not os.path.exists(TRAIN_CSV):
        raise RuntimeError(f"Training CSV not found at {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    df_use = df[FEATURE_COLS + [LABEL_COL]].dropna().copy()
    feats = [_row_to_feature(r) for r in df_use.to_dict("records")]
    fc = ee.FeatureCollection(feats).randomColumn('rand', 42)
    train_fc = fc.filter(ee.Filter.lt('rand', 0.8))
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
    bounds: Dict[str, float]   # {minx,miny,maxx,maxy}
    width: int
    height: int
    crs: str
    message: str

class ResolveAreaBody(BaseModel):
    named_area: dict  # { level, name, country }

class ResolveAreaOut(BaseModel):
    bounds: Dict[str, float]
    geojson: dict  # pure Geometry JSON

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
        classified = pred_img.classify(CLASSIFIER).rename("flood_pred")  # 0 or 1

        flood_only = classified.eq(1).selfMask()
        vis_img = flood_only.visualize(min=0, max=1, palette=["ff0000"])

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

        alpha = rgba[:, :, 3]
        flooded = alpha > 0
        out_arr = np.zeros((height, width), dtype=np.uint8)
        out_arr[flooded] = 1  # 1 = flood

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
            dst.write(out_arr, 1)

        base = str(request.base_url)  # e.g. http://localhost:8000/
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
