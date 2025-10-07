import { projects } from "../data";
import ProjectCard from "./ProjectCard";
export default function Projects(){
  return (
    <section id="projects" className="section bg-slate-50/70">
      <div className="container-xl">
        <h2 className="text-2xl font-bold mb-6">Projects</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {projects.map(p => <ProjectCard key={p.title} {...p} />)}
        </div>
      </div>
    </section>
  )
}
