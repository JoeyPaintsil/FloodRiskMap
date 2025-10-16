// src/pages/FloodMapper.jsx
import { Link } from "react-router-dom";
import { ArrowLeft, BarChart3 } from "lucide-react";

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
              src="/hero-map.png"  /* replace with a flood image or map screenshot */
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
              An automated system that predicts flood-prone areas using global Earth
              observation data and machine learning, so planners can act before waters rise.
            </p>

            <ul className="space-y-3 text-slate-700 mb-8">
              <li className="pl-2">
                • <strong>1.81 billion people</strong> live in areas exposed to
                significant flooding in a 1-in-100-year event, most in low- and middle-income
                countries. <span className="text-slate-500 text-sm">(World Bank)</span>
              </li>
              <li className="pl-2">
                • Global flood exposure has been rising for decades, with the largest losses in Asia. 
                <span className="text-slate-500 text-sm"> (UNDRR)</span>
              </li>
              <li className="pl-2">
                • Disasters push <strong>~26 million people into poverty each year</strong> — floods and storms are major drivers. 
                <span className="text-slate-500 text-sm"> (World Bank)</span>
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

        {/* Keep the rest of your page (features, preview, contact CTA) below if you like */}
      </div>
    </section>
  );
}
