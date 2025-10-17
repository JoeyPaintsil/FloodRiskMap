// src/pages/FloodMapper.jsx
import { Link } from "react-router-dom";
import { ArrowLeft, BarChart3, Search, Upload, Brain, Download } from "lucide-react";

export default function FloodMapper() {
  return (
    <section className="section bg-slate-50 min-h-screen">
      <div className="container-xl">
        {/* Back to Home */}
        <div className="mb-6">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-slate-200 text-slate-700 hover:bg-brand-600 hover:text-white transition"
          >
            <ArrowLeft size={16} />
            Back to Home
          </Link>
        </div>

        {/* Hero: image left, text right */}
        <div className="grid md:grid-cols-2 gap-10 items-center mb-12">
          {/* Left: image */}
          <div className="card overflow-hidden">
            <img
              src="/flood_map.jpg"
              alt="Flood risk map preview"
              className="w-full h-full object-cover"
            />
          </div>

          {/* Right: text */}
          <div>
            <h1 className="text-4xl md:text-5xl font-extrabold mb-4 text-brand-700">
              Flood JoeMapper
            </h1>
            <p className="text-lg text-slate-700 mb-6">
              Flooding is one of the most frequent and damaging natural hazards worldwide, 
              affecting millions of people and causing massive economic losses.{" "}
              <strong>Flood JoeMapper</strong> helps decision-makers identify and predict flood-prone 
              regions anywhere in the world using Earth observation data and machine learning.
            </p>

            <ul className="space-y-3 text-slate-700 mb-8">
              <li className="pl-2">
                • <strong>1.81 billion people</strong> live in areas exposed to significant flooding in a 
                1-in-100-year event, most in low- and middle-income countries.{" "}
                <span className="text-slate-500 text-sm">(World Bank)</span>
              </li>
              <li className="pl-2">
                • Climate variability and rapid urban growth have increased exposure and vulnerability 
                in many low- and middle-income regions.{" "}
                <span className="text-slate-500 text-sm">(UNDRR)</span>
              </li>
              <li className="pl-2">
                • Flood risk prediction tools like <strong>Flood JoeMapper</strong> enable faster preparedness, 
                improved resource allocation, and better mitigation strategies before disaster strikes.{" "}
                <span className="text-slate-500 text-sm">(World Bank)</span>
              </li>
            </ul>

            <Link
              to="/flood-mapper/dashboard"
              className="inline-flex items-center gap-2 rounded-md px-5 py-3 bg-brand-600 text-white font-medium hover:bg-brand-700 transition"
            >
              <BarChart3 size={18} />
              Open Dashboard
            </Link>
          </div>
        </div>

        {/* How the Dashboard Works */}
        <section aria-labelledby="how-it-works">
          <h2 id="how-it-works" className="text-2xl md:text-3xl font-bold mb-6">
            How the Dashboard Works
          </h2>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* 1 */}
            <div className="card p-5">
              <div className="flex items-center gap-3 mb-3">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-700 font-semibold">1</span>
                <span className="text-slate-800 font-medium">Choose Your Area</span>
              </div>
              <div className="flex items-center gap-2 text-blue-600 mb-2">
                <Search size={18} /> <Upload size={18} />
              </div>
              <p className="text-slate-600 text-sm">
                Search by place name or upload a <code>.geojson</code> file to set your Area of Interest (AOI).
              </p>
            </div>

            {/* 2 */}
            <div className="card p-5">
              <div className="flex items-center gap-3 mb-3">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-700 font-semibold">2</span>
                <span className="text-slate-800 font-medium">Run Prediction</span>
              </div>
              <div className="flex items-center gap-2 text-blue-600 mb-2">
                <BarChart3 size={18} />
              </div>
              <p className="text-slate-600 text-sm">
                Click <strong>Predict Flood Risk</strong>. The system prepares inputs and starts the analysis for your AOI.
              </p>
            </div>

            {/* 3 */}
            <div className="card p-5">
              <div className="flex items-center gap-3 mb-3">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-700 font-semibold">3</span>
                <span className="text-slate-800 font-medium">Model & Results</span>
              </div>
              <div className="flex items-center gap-2 text-blue-600 mb-2">
                <Brain size={18} />
              </div>
              <p className="text-slate-600 text-sm">
                Behind the scenes, ML models combine EO data and terrain factors to map areas with higher flood risk within your AOI.
              </p>
            </div>

            {/* 4 */}
            <div className="card p-5">
              <div className="flex items-center gap-3 mb-3">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-700 font-semibold">4</span>
                <span className="text-slate-800 font-medium">Download Your Map</span>
              </div>
              <div className="flex items-center gap-2 text-blue-600 mb-2">
                <Download size={18} />
              </div>
              <p className="text-slate-600 text-sm">
                Export the flood risk map as <code>.png</code> or GeoTIFF (<code>.tif</code>) and continue analysis in GIS tools like ArcGIS or QGIS.
              </p>
            </div>
          </div>
        </section>
      </div>
    </section>
  );
}
