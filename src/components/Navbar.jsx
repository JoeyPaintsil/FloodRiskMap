import { useState } from "react";
import { Menu, X } from "lucide-react";

export default function Navbar(){
  const [open, setOpen] = useState(false);
  const nav = [
    {href:"#about", label:"About"},
    {href:"#skills", label:"Skills"},
    {href:"#projects", label:"Projects"},
    {href:"#experience", label:"Experience"},
    {href:"#contact", label:"Contact"},
  ];
  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur border-b border-slate-200">
      <div className="container-xl flex items-center justify-between h-16">
        <a href="#" className="font-bold">JP</a>
        <nav className="hidden md:flex items-center gap-6">
          {nav.map(n => <a key={n.href} href={n.href} className="hover:text-brand-700">{n.label}</a>)}
        </nav>
        <button className="md:hidden btn" onClick={()=>setOpen(v=>!v)} aria-label="Toggle menu">
          {open ? <X size={18}/> : <Menu size={18}/>}
        </button>
      </div>
      {open && (
        <div className="md:hidden border-t border-slate-200 bg-white">
          <div className="container-xl py-4 flex flex-col gap-3">
            {nav.map(n => <a key={n.href} href={n.href} className="py-1">{n.label}</a>)}
          </div>
        </div>
      )}
    </header>
  )
}
