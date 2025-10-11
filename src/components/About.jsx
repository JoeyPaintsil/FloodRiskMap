// src/components/About.jsx
import { profile, stats } from "../data";

export default function About() {
  return (
    <section id="about" className="section bg-white">
      <div className="container-xl grid md:grid-cols-2 gap-10 items-center">
        {/* Left: portrait */}
        <div className="relative">
          <div className="aspect-[4/5] w-full overflow-hidden rounded-2xl border border-slate-200 shadow-soft bg-slate-100">
            <img
              src={profile.photo}
              alt={`${profile.name} portrait`}
              className="h-full w-full object-cover object-[50%_8%]"
            />
          </div>
        </div>

        {/* Right: text and stats */}
        <div>
          <div className="inline-flex items-center gap-2">
            <div className="h-[2px] w-6 bg-orange-500"></div>
            <span className="text-sm font-semibold text-orange-600">About Me</span>
          </div>

          <h2 className="mt-4 text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900">
            GIS, Remote Sensing and Web Development
          </h2>

          <p className="mt-3 text-2xl font-semibold text-slate-500">
            Exploring data, maps, and web technology to make complex information easier to see and use.
          </p>

          <p className="mt-5 text-slate-700 leading-relaxed text-justify">
            I design end-to-end geospatial workflows using Google Earth Engine, Python, and machine learning, 
            and I also build websites and interactive geospatial dashboards that bring spatial insights to life. 
            My focus is on reproducibility, performance, and clear communication. 
            I’m currently working at{" "}
            <u><a
              href="https://www.abeya.co/"
              target="_blank"
              rel="noreferrer"
              className="text-brand-700 font-semibold hover:underline"
            >

              Abeya
            </a></u>{" "}
            
            as a Remote Sensing Data Scientist. 
            I enjoy connecting with others who are passionate about geospatial data, machine learning, and web development. 
            Feel free to reach out or connect with me on LinkedIn.
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
