// src/pages/FloodDashboard.jsx
import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import {
  Layers,
  Target,
  HelpCircle,
  Upload,
  Plus,
  Home,
  Search,
  Download,
  Eye,
  EyeOff,
} from "lucide-react";
import { Link } from "react-router-dom";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/** Debounce hook */
function useDebounce(value, ms = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export default function FloodDashboard() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const fileRef = useRef(null);

  // Search
  const [query, setQuery] = useState("");
  const debounced = useDebounce(query, 300);
  const [items, setItems] = useState([]);
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);

  // CSV contents
  const [csvRows, setCsvRows] = useState([]);

  // AOI / Named-area selection
  const aoiGeoJsonRef = useRef(null);   // raw uploaded GeoJSON
  const selectedAreaRef = useRef(null); // { level: "district"|"region", name, country }

  // Progress & result
  const [status, setStatus] = useState("");  // "", "running", "done", "error"
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState("");
  const [result, setResult] = useState(null);

  // Overlay toggle
  const [floodVisible, setFloodVisible] = useState(true);

  // Basemap
  const [basemap, setBasemap] = useState("osm"); // 'osm' | 'sat' | 'topo'

  // ---------------- Map init ----------------
  useEffect(() => {
    const style = buildStyle(basemap);
    const map = new maplibregl.Map({
      container: mapRef.current,
      style,
      center: [-0.19, 5.55], // Ghana-ish
      zoom: 10,
    });

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

    mapInstanceRef.current = map;
    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Basemap switcher (rebuild style but keep overlays)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const style = buildStyle(basemap);
    map.setStyle(style);

    // Restore overlays after style load
    map.once("styledata", () => {
      if (aoiGeoJsonRef.current) addOrUpdateAOILayers(map, aoiGeoJsonRef.current, "aoi");
      if (map.__namedAOI) addOrUpdateAOILayers(map, map.__namedAOI, "named-aoi");
      if (map.__floodOverlay) addFloodOverlay(map, map.__floodOverlay);
    });
  }, [basemap]);

  // ---------------- Load CSV once ----------------
  useEffect(() => {
    fetch("/data/GAUL_country_region_district.csv")
      .then((r) => r.text())
      .then((txt) => {
        const lines = txt.split(/\r?\n/).filter(Boolean);
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
          const parts = lines[i].split(",");
          if (!parts.length) continue;
          const Country = (parts[0] ?? "").trim();
          const Region  = (parts[1] ?? "").trim();
          const District= (parts[2] ?? "").trim();
          if (Country) rows.push({ country: Country, region: Region, district: District });
        }
        setCsvRows(rows);
      })
      .catch(() => setCsvRows([]));
  }, []);

  // ---------------- Search → suggestions from CSV ----------------
  useEffect(() => {
    const q = debounced.trim().toLowerCase();
    if (q.length < 2) {
      setItems([]);
      setOpen(false);
      return;
    }

    const qLeft = q.split(",")[0].trim();

    // Try District first
    const districtHits = csvRows
      .filter((r) => r.district && r.district.toLowerCase().includes(qLeft))
      .slice(0, 5)
      .map((r) => ({
        label: `${r.district}, ${r.country}`,
        level: "district",
        name: r.district,
        country: r.country,
      }));

    if (districtHits.length > 0) {
      setItems(districtHits);
      setOpen(true);
      return;
    }

    // Fallback: Region
    const regionHits = csvRows
      .filter((r) => r.region && r.region.toLowerCase().includes(qLeft))
      .slice(0, 5)
      .map((r) => ({
        label: `${r.region}, ${r.country}`,
        level: "region",
        name: r.region,
        country: r.country,
      }));

    setItems(regionHits);
    setOpen(regionHits.length > 0);
  }, [debounced, csvRows]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (!dropdownRef.current) return;
      if (!dropdownRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // ---------------- Upload AOI ----------------
  const onUploadClick = () => fileRef.current?.click();

  function unwrapGeometry(geo) {
    if (!geo) return null;
    if (geo.type === "FeatureCollection") return geo.features?.[0]?.geometry ?? null;
    if (geo.type === "Feature") return geo.geometry ?? null;
    return geo;
  }

  function collectRings(geoOrGeom) {
    const geom = unwrapGeometry(geoOrGeom);
    if (!geom) return [];
    switch (geom.type) {
      case "Polygon": return geom.coordinates;
      case "MultiPolygon": return geom.coordinates.flat();
      case "LineString": return [geom.coordinates];
      case "MultiLineString": return geom.coordinates;
      case "Point": return [[geom.coordinates]];
      case "MultiPoint": return [geom.coordinates];
      case "GeometryCollection": return (geom.geometries || []).flatMap((g) => collectRings(g));
      default: return [];
    }
  }

  const onFileChosen = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const geo = JSON.parse(text);
      const map = mapInstanceRef.current;
      if (!map) return;

      // Uploaded AOI takes precedence; clear named-area selection
      aoiGeoJsonRef.current = geo;
      selectedAreaRef.current = null;

      // Show filename in search box
      setQuery(`AOI Uploaded: ${file.name}`);
      setOpen(false);

      addOrUpdateAOILayers(map, geo, "aoi");

      // Fit to AOI bounds
      const bbox = computeBBox(geo);
      if (bbox) {
        map.fitBounds(bbox, { padding: 60, duration: 800 });
      }
    } catch (err) {
      console.error(err);
      alert("Could not read this file. Please upload a valid GeoJSON (.geojson or .json).");
    } finally {
      e.target.value = ""; // allow re-choosing the same file
    }
  };

  // Compute [minX, minY, maxX, maxY] for GeoJSON
  function computeBBox(geo) {
    const coords = [];
    const pushCoords = (c) => {
      if (Array.isArray(c) && typeof c[0] === "number" && typeof c[1] === "number") {
        coords.push(c);
      } else if (Array.isArray(c)) {
        c.forEach(pushCoords);
      }
    };
    if (geo.type === "FeatureCollection") geo.features.forEach((f) => pushCoords(f.geometry?.coordinates));
    else if (geo.type === "Feature") pushCoords(geo.geometry?.coordinates);
    else if (geo.type && geo.coordinates) pushCoords(geo.coordinates);

    if (!coords.length) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    coords.forEach(([x, y]) => {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    });
    return [[minX, minY], [maxX, maxY]];
  }

  // ---------------- Choose suggestion (GAUL area) ----------------
  const choose = async (it) => {
    const named = { level: it.level, name: it.name, country: it.country };
    selectedAreaRef.current = named;
    aoiGeoJsonRef.current = null; // named-area takes precedence
    setQuery(it.label);
    setOpen(false);

    // Ask backend for the GAUL geometry to draw an outline immediately
    try {
      const res = await fetch(`${API_URL}/resolve_area`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ named_area: named }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      const map = mapInstanceRef.current;
      if (!map) return;

      const geo = {
        type: "Feature",
        properties: {},
        geometry: data.geojson, // unioned geometry from backend
      };
      map.__namedAOI = geo; // remember so we can restore after basemap switch
      addOrUpdateAOILayers(map, geo, "named-aoi");

      const { minx, miny, maxx, maxy } = data.bounds;
      map.fitBounds([[minx, miny], [maxx, maxy]], { padding: 60, duration: 800 });
    } catch (err) {
      console.error(err);
    }
  };

  // ------------ Map helpers ------------
  function addOrUpdateAOILayers(map, geo, idPrefix) {
    const srcId = `${idPrefix}-src`;
    const fillId = `${idPrefix}-fill`;
    const lineId = `${idPrefix}-outline`;

    if (map.getSource(srcId)) {
      map.getSource(srcId).setData(geo);
    } else {
      map.addSource(srcId, { type: "geojson", data: geo });
      map.addLayer({
        id: fillId,
        type: "fill",
        source: srcId,
        paint: { "fill-color": "#22c55e", "fill-opacity": 0.12 },
      });
      map.addLayer({
        id: lineId,
        type: "line",
        source: srcId,
        paint: { "line-color": "#16a34a", "line-width": 2 },
      });
    }
  }

  function addFloodOverlay(map, payload) {
    const { url, corners } = payload;
    if (map.getLayer("flood_overlay_layer")) map.removeLayer("flood_overlay_layer");
    if (map.getSource("flood_overlay")) map.removeSource("flood_overlay");
    map.addSource("flood_overlay", { type: "image", url, coordinates: corners });
    map.addLayer({ id: "flood_overlay_layer", type: "raster", source: "flood_overlay", paint: { "raster-opacity": 1.0 } });
  }

  function buildStyle(which) {
    const sources = {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: '&copy; OpenStreetMap contributors',
      },
      topo: {
        type: "raster",
        tiles: ["https://tile.opentopomap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: '&copy; OpenTopoMap & OSM',
      },
      sat: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: '&copy; Esri WorldImagery',
      },
    };
    const src = which === "sat" ? "sat" : which === "topo" ? "topo" : "osm";
    return {
      version: 8,
      sources: { base: sources[src] },
      layers: [{ id: "base", type: "raster", source: "base" }],
    };
  }

  // ---------------- Predict ----------------
  const onPredict = async () => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const body = {};
    if (aoiGeoJsonRef.current) {
      const rings = collectRings(aoiGeoJsonRef.current);
      if (rings && rings.length) body.aoi_coords = rings[0];
    } else if (selectedAreaRef.current) {
      body.named_area = selectedAreaRef.current; // {level,name,country}
    } else {
      alert("Pick a Region/District from suggestions or upload an AOI first.");
      return;
    }

    // Reset progress
    setResult(null);
    setStatus("running");
    setProgress(12);
    setStep("Preparing request…");
    setFloodVisible(true);

    try {
      setProgress(30);
      setStep("Calling API…");

      const resp = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || `HTTP ${resp.status}`);
      }

      const data = await resp.json();

      setProgress(72);
      setStep("Rendering overlay…");

      const { minx, miny, maxx, maxy } = data.bounds;
      const corners = [
        [minx, maxy], // top-left
        [maxx, maxy], // top-right
        [maxx, miny], // bottom-right
        [minx, miny], // bottom-left
      ];

      // Save to restore after basemap change
      map.__floodOverlay = { url: data.png_url, corners };
      addFloodOverlay(map, map.__floodOverlay);

      map.fitBounds([[minx, miny], [maxx, maxy]], { padding: 60, duration: 800 });

      setResult(data);
      setProgress(100);
      setStep("Prediction complete");
      setStatus("done");

      // Auto-hide progress in 15s
      setTimeout(() => {
        setStatus("");
        setProgress(0);
        setStep("");
      }, 15000);
    } catch (err) {
      console.error(err);
      setStatus("error");
      setStep("Prediction failed");
      setProgress(0);
      alert(err.message || "Prediction failed. Check API logs.");
    }
  };

  // Toggle overlay visibility
  const toggleFloodLayer = () => {
    const map = mapInstanceRef.current;
    if (!map || !map.getLayer("flood_overlay_layer")) return;
    const vis = floodVisible ? "none" : "visible";
    map.setLayoutProperty("flood_overlay_layer", "visibility", vis);
    setFloodVisible(!floodVisible);
  };

  // Cycle basemap
  const cycleBasemap = () =>
    setBasemap((b) => (b === "osm" ? "sat" : b === "sat" ? "topo" : "osm"));

  return (
    <section className="relative h-[100vh] bg-slate-900">
      {/* Map */}
      <div ref={mapRef} className="absolute inset-0" />

      {/* Back to Home (desktop) */}
      <div className="absolute top-4 left-4 z-30 hidden md:block">
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-slate-800/90 text-slate-200 hover:bg-slate-700 backdrop-blur"
        >
          <Home size={16} /> Back to Home
        </Link>
      </div>

      {/* Top toolbar */}
      <div className="absolute top-2 md:top-4 left-1/2 -translate-x-1/2 z-30 w-[calc(100vw-16px)] md:w-[calc(100vw-240px)] lg:w-[min(1200px,calc(100vw-360px))]">
        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2 md:gap-3">
          {/* Mobile Home */}
          <Link
            to="/"
            className="md:hidden order-1 inline-flex items-center gap-2 px-3 py-2 rounded-md bg-slate-800/90 text-white hover:bg-slate-700 backdrop-blur"
          >
            <Home size={16} />
            <span>Home</span>
          </Link>

          {/* Search + dropdown */}
          <div className="relative flex-1 order-2" ref={dropdownRef}>
            <div className="flex items-center gap-2 rounded-xl bg-white/95 px-3 md:px-4 py-2 shadow-soft">
              <Search size={16} className="text-slate-500 shrink-0" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={() => query.length >= 2 && setOpen(true)}
                type="text"
                placeholder="Search Region/District (GAUL) or Upload AOI"
                className="w-full bg-transparent outline-none text-slate-800 placeholder:text-slate-400"
              />
            </div>

            {open && (
              <div className="absolute z-40 mt-1 w-full rounded-xl bg-white shadow-soft max-h-72 overflow-auto">
                {items.length === 0 ? (
                  <div className="px-4 py-3 text-sm text-slate-600">
                    No area matches your search.{" "}
                    <button
                      className="underline"
                      onClick={() => {
                        setOpen(false);
                        onUploadClick();
                      }}
                    >
                      Try uploading an AOI
                    </button>
                    .
                  </div>
                ) : (
                  items.map((it, i) => (
                    <button
                      key={`${it.label}-${i}`}
                      onClick={() => choose(it)}
                      className="w-full text-left px-4 py-2 hover:bg-slate-100 flex items-center justify-between"
                    >
                      <span>{it.label}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-slate-200 text-slate-700">
                        {it.level === "district" ? "District (GAUL)" : "Region (GAUL)"}
                      </span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Predict */}
          <div className="order-3">
            <button
              onClick={onPredict}
              disabled={status === "running"}
              className="inline-flex items-center gap-2 px-4 md:px-5 py-2 rounded-md bg-brand-600 text-white font-medium hover:bg-brand-700 shadow-soft disabled:opacity-60 disabled:cursor-not-allowed"
            >
              Predict Flood Risk
            </button>
          </div>
        </div>
      </div>

      {/* Progress pill — small; no fullscreen overlay */}
      {(status === "running" || status === "done") && (
        <div className="pointer-events-none absolute md:top-20 top-auto bottom-24 left-1/2 -translate-x-1/2 z-30">
          <div className="bg-slate-800/95 text-white px-3 md:px-4 py-2 rounded-lg shadow-soft flex items-center gap-2 md:gap-3 max-w-[min(90vw,520px)]">
            <div className="w-36 md:w-48 h-2 bg-slate-600 rounded">
              <div className="h-2 bg-emerald-400 rounded" style={{ width: `${progress}%` }} />
            </div>
            <span className="text-xs md:text-sm whitespace-nowrap">{step || "Working..."}</span>
            <span className="text-xs md:text-sm opacity-80">{progress}%</span>
          </div>
        </div>
      )}

      {/* Legend + toggle — left-top desktop, bottom-left mobile */}
      {result && (
        <div className="absolute z-30 md:left-4 md:top-24 left-2 right-auto top-auto bottom-36">
          <div className="flex md:flex-col gap-2">
            <div className="rounded-md bg-white/95 p-2 md:p-3 shadow-soft">
              <div className="font-medium text-slate-700 mb-1 text-xs md:text-sm">Legend</div>
              <div className="flex items-center gap-2 text-xs md:text-sm text-slate-700">
                <span className="inline-block w-3 h-3 md:w-4 md:h-4 rounded-sm" style={{ background: "#ff0000" }} />
                Flood risk
              </div>
            </div>

            <button
              onClick={toggleFloodLayer}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-white text-slate-800 shadow-soft text-xs md:text-sm"
              title={floodVisible ? "Hide Flood Layer" : "Show Flood Layer"}
            >
              {floodVisible ? <EyeOff size={16} /> : <Eye size={16} />}{" "}
              {floodVisible ? "Hide Flood Layer" : "Show Flood Layer"}
            </button>
          </div>
        </div>
      )}

      {/* Download buttons — top-right desktop, bottom-center mobile */}
      {result && (
        <div className="absolute z-30 md:top-20 md:right-4 top-auto bottom-4 left-0 right-0 md:left-auto flex md:items-end justify-center md:justify-end gap-2">
          <div className="hidden md:block text-emerald-300 text-sm font-medium mr-1">Prediction complete</div>
          <a
            href={result.png_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-white text-slate-800 shadow-soft text-xs md:text-sm"
          >
            <Download size={16} /> PNG
          </a>
          <a
            href={result.tif_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-white text-slate-800 shadow-soft text-xs md:text-sm"
          >
            <Download size={16} /> GeoTIFF
          </a>
        </div>
      )}

      {/* Right-side round controls */}
      <div className="absolute right-2 md:right-4 bottom-28 md:bottom-24 z-20 flex flex-col gap-2 md:gap-3">
        <button className="p-3 rounded-full bg-white/90 hover:bg-white shadow-soft" title="Locate">
          <Target size={18} />
        </button>
        <button
          className="p-3 rounded-full bg-white/90 hover:bg-white shadow-soft"
          title={`Basemap: ${basemap === "osm" ? "OSM" : basemap === "sat" ? "Satellite" : "Topo"}`}
          onClick={() => setBasemap((b) => (b === "osm" ? "sat" : b === "sat" ? "topo" : "osm"))}
        >
          <Layers size={18} />
        </button>
        <button className="p-3 rounded-full bg-white/90 hover:bg-white shadow-soft" title="Help">
          <HelpCircle size={18} />
        </button>
      </div>

      {/* Upload AOI */}
      <div className="absolute right-2 md:right-4 bottom-4 z-30">
        <button
          onClick={onUploadClick}
          className="inline-flex items-center gap-2 px-3 md:px-4 py-2 rounded-md bg-slate-800/90 text-slate-100 hover:bg-slate-700 backdrop-blur text-xs md:text-sm"
          title="Upload AOI"
        >
          <Upload size={16} /> Upload AOI
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".geojson,.json"
          className="hidden"
          onChange={onFileChosen}
        />
      </div>

      {/* Bottom status bar (desktop only) */}
      <div className="absolute left-4 right-4 bottom-4 z-20 hidden lg:flex items-center justify-between text-slate-200">
        <div className="text-sm opacity-80">Flood risk prototype - demo data. Replace with your model outputs.</div>
        <button className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-slate-800/90 text-slate-100 hover:bg-slate-700">
          <Plus size={16} /> Add Layer
        </button>
      </div>
    </section>
  );
}
