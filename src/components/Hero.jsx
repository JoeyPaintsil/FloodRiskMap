// src/components/Hero.jsx
import { motion } from "framer-motion";
import { profile } from "../data";
import * as Icons from "lucide-react";

export default function Hero() {
  return (
    <section className="section">
      <div className="container-xl grid md:grid-cols-2 gap-10 items-center">
        {/* Left: text */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-5 md:space-y-6"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 leading-snug">
            <span className="block">
              Hello  I'm
            </span>
            <span className="block">
             <span className="block mt-2 text-brand-700">{profile.name}</span>
            </span>
          </h1>

          {/* use blurb from profile */}
          <p className="text-lg text-slate-600 max-w-[46ch] text-justify">{profile.blurb}</p>

          <div className="flex flex-wrap gap-3">
            <a href="/flood-mapper" className="btn btn-primary">View my latest project</a>
            <a href="#contact" className="btn">Contact Me</a>
          </div>

          <div className="flex flex-wrap gap-3">
            {profile.socials.map((s) => {
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

        {/* Map visual */}
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="card p-0 overflow-hidden aspect-square md:aspect-[4/3]"
        >
          <img src="/hero-map.png" alt="Map preview" className="w-full h-full object-cover" />
        </motion.div>
      </div>
    </section>
  );
}
