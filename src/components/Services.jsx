import { services } from "../data";
import { Map, Cpu, Server, BarChart3, ArrowRight } from "lucide-react";

const ICONS = { Map, Cpu, Server, BarChart3 };

export default function Services(){
  return (
    <section id="services" className="section bg-slate-50/70">
      <div className="container-xl">
        {/* Section pretitle + title */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2">
            <div className="h-[2px] w-6 bg-brandBlue-500"></div>
            <span className="text-sm font-semibold text-brand-600">Core Expertise</span>
          </div>
          <h2 className="mt-4 text-4xl md:text-5xl font-extrabold">Focus Areas</h2>
        </div>

        {/* Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {services.map((s) => {
            const Icon = ICONS[s.icon] ?? Map;
            return (
              <article key={s.title} className="card p-8 group flex flex-col justify-between">
                {/* Icon + soft circle */}
                <div className="relative mb-6">
                  <div className="absolute left-2 top-1 h-10 w-10 rounded-full bg-slate-200/50" />
                  <Icon className="relative z-10 text-brand-600" size={36} />
                </div>

                <h3 className="text-xl font-semibold mb-2">{s.title}</h3>
                <p className="text-slate-600 mb-6">{s.desc}</p>

                {/* Bottom arrow button */}
                <div className="mt-auto">
                  <a
                    href="#projects"
                    className="inline-flex items-center gap-2 rounded-md px-3 py-2 bg-slate-200 text-slate-700 group-hover:bg-brand-500 group-hover:text-white transition"
                  >
                    <ArrowRight size={16} />
                  </a>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
