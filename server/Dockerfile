FROM python:3.11-slim

# If you need GDAL/rasterio compiled, keep these. Otherwise you can remove libgdal-dev.
RUN apt-get update && apt-get install -y \
    gdal-bin gdal-data libgdal-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt is in this same server/ folder
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ⬇️ FIX: copy the current folder into /app/server (not "server/" from parent)
COPY . /app/server/

# Start FastAPI
CMD uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}
