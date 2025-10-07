export default function ProjectCard({title, year, tags, description, links}){
  return (
    <div className="card p-6 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold">{title}</h3>
        <span className="text-sm text-slate-500">{year}</span>
      </div>
      <p className="text-slate-600">{description}</p>
      <div className="flex flex-wrap gap-2">
        {tags.map(t => <span key={t} className="chip">{t}</span>)}
      </div>
      <div className="pt-2 flex gap-3">
        {links?.map(l => <a key={l.label} href={l.href} target="_blank" rel="noreferrer" className="btn">{l.label}</a>)}
      </div>
    </div>
  )
}
