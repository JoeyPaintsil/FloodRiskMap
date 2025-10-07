import { experience } from "../data";
export default function Experience(){
  return (
    <section id="experience" className="section">
      <div className="container-xl">
        <h2 className="text-2xl font-bold mb-6">Experience</h2>
        <div className="space-y-4">
          {experience.map((e) => (
            <div key={e.org} className="card p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold">{e.role}</h3>
                  <p className="text-slate-600">{e.org}</p>
                </div>
                <span className="text-sm text-slate-500">{e.period}</span>
              </div>
              <ul className="list-disc pl-5 mt-3 text-slate-700 space-y-1">
                {e.bullets.map((b,i) => <li key={i}>{b}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
