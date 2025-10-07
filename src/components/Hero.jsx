// src/components/Hero.jsx
import { motion } from "framer-motion";
import { profile } from "../data";
import * as Icons from "lucide-react";

export default function Hero(){
  return (
    <section className="section">
      <div className="container-xl grid md:grid-cols-2 gap-10 items-center">
        {/* Left: text */}
        <motion.div
          initial={{opacity:0, y:10}}
          animate={{opacity:1, y:0}}
          transition={{duration:.6}}
          className="space-y-6"
        >
          {/* <div className="inline-flex items-center gap-2 chip">
            <span className="w-2 h-2 rounded-full bg-green-500"></span>
            Available for projects
          </div> */}

          {/* Name */}
          <h1 className="text-2xl md:text-3xl font-semibold text-brand-700">
            {profile.name}
          </h1>

          {/* Role / Title */}
          <h2 className="text-4xl md:text-5xl font-extrabold leading-tight text-slate-900">
            Remote Sensing Data Scientist
          </h2>


          <p className="text-lg text-slate-600">{profile.blurb}</p>

          {/* Primary actions */}
          <div className="flex flex-wrap gap-3">
           <a href="/flood-mapper" className="btn btn-primary">
              View my latest project!
          </a>

            <a href="#contact" className="btn">
              Contact Me
            </a>
          </div>

          {/* Socials with icons */}
          <div className="flex flex-wrap gap-3">
            {profile.socials.map(s => {
              const Icon = Icons[s.icon] || Icons.Circle;
              return (
                <a
                  key={s.label}
                  href={s.href}
                  target="_blank"
                  rel="noreferrer"
                  className="chip hover:bg-slate-200 inline-flex items-center gap-2"
                >
                  <Icon size={16} className="text-slate-600" />
                  <span>{s.label}</span>
                </a>
              );
            })}
          </div>
        </motion.div>

        {/* Right: visual */}
        <motion.div
          initial={{opacity:0, scale:.96}}
          animate={{opacity:1, scale:1}}
          transition={{duration:.6, delay:.1}}
          className="card p-6 aspect-square md:aspect-[4/3] flex items-center justify-center"
        >
          <img src="/hero-map.png" alt="Map preview" className="rounded-xl object-cover" />
        </motion.div>
      </div>
    </section>
  );
}
