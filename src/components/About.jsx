// src/components/About.jsx
import { profile, stats } from "../data";

export default function About(){
  return (
    <section id="about" className="section bg-white">
      <div className="container-xl grid md:grid-cols-2 gap-10 items-center">
        {/* Left: portrait with overlay badge */}
        <div className="relative">
          <div className="aspect-[4/5] w-full overflow-hidden rounded-2xl border border-slate-200 shadow-soft bg-slate-100">
            <img
              src={profile.photo}
              alt={`${profile.name} portrait`}
              className="h-full w-full object-cover"
            />
          </div>

          {/* Years overlay card */}
          <div className="absolute -bottom-6 left-6">
            <div className="rounded-2xl bg-slate-900/90 text-white px-6 py-5 shadow-xl">
              <div className="text-4xl font-extrabold leading-none">{profile.years}</div>
              <div className="text-sm opacity-90 mt-1">Years Experience</div>
            </div>
          </div>
        </div>

        {/* Right: text and stats */}
        <div>
          <div className="inline-flex items-center gap-2">
            <div className="h-[2px] w-6 bg-orange-500"></div>
            <span className="text-sm font-semibold text-orange-600">About Me</span>
          </div>

          <h2 className="mt-4 text-4xl md:text-5xl font-extrabold tracking-tight">
            Welcome — GIS and Remote Sensing Services
          </h2>

          <p className="mt-3 text-2xl font-semibold text-slate-500">
            {profile.subhead}
          </p>

          <p className="mt-5 text-slate-600 leading-relaxed">
            I design end-to-end geospatial workflows using Google Earth Engine, Python, and
            machine learning to deliver reliable maps and analytics for forestry, agriculture,
            risk and resilience. My focus is on reproducibility, performance, and clear
            communication so stakeholders can act with confidence.
          </p>

          {/* Stats */}
          <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-6">
            {stats.map((s) => (
              <div key={s.label}>
                <div className="text-3xl font-extrabold text-orange-600">{s.value}</div>
                <div className="text-sm text-slate-500 mt-1">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
