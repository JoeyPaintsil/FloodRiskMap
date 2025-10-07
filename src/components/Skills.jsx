import { skillCategories, certifications, currentlyLearning } from "../data";
import * as Icons from "lucide-react";

export default function Skills() {
  return (
    <section id="skills" className="section bg-slate-50/70">
      <div className="container-xl">
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2">
            <div className="h-[2px] w-6 bg-brandOrange-500"></div>
            <span className="text-sm font-semibold text-brandOrange-600">Skills</span>
          </div>
          <h2 className="mt-4 text-4xl md:text-5xl font-extrabold">My Professional Toolkit</h2>
        </div>

        {/* Categories */}
        <div className="grid md:grid-cols-2 gap-6">
          {skillCategories.map(cat => (
            <div key={cat.name} className="card p-6">
              <h3 className="text-lg font-semibold mb-4">{cat.name}</h3>
              <div className="grid sm:grid-cols-2 gap-3">
                {cat.items.map(skill => {
                  const Icon = Icons[skill.icon] || Icons.Circle;
                  return (
                    <div key={skill.name} className="flex items-center gap-2">
                      <Icon size={18} className="text-brandOrange-600" />
                      <span>{skill.name}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Extras */}
        <div className="mt-10 grid md:grid-cols-2 gap-6">
          <div className="card p-6">
            <h4 className="text-lg font-semibold mb-3">Certifications</h4>
            <div className="flex flex-wrap gap-2">
              {certifications.map(c => (
                <span key={c} className="chip">{c}</span>
              ))}
            </div>
          </div>
          <div className="card p-6">
            <h4 className="text-lg font-semibold mb-3">Currently Learning</h4>
            <div className="flex flex-wrap gap-2">
              {currentlyLearning.map(c => (
                <span key={c} className="chip">{c}</span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
