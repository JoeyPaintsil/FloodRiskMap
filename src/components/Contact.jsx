export default function Contact(){
  return (
    <section id="contact" className="section bg-slate-50/70">
      <div className="container-xl grid md:grid-cols-3 gap-8 items-start">
        <div className="md:col-span-1">
          <h2 className="text-2xl font-bold">Contact</h2>
        </div>
        <div className="md:col-span-2 card p-6">
          <form className="grid gap-4">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-600">Name</label>
                <input type="text" className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300" placeholder="Your name"/>
              </div>
              <div>
                <label className="text-sm text-slate-600">Email</label>
                <input type="email" className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300" placeholder="you@example.com"/>
              </div>
            </div>
            <div>
              <label className="text-sm text-slate-600">Message</label>
              <textarea rows="5" className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300" placeholder="Tell me about your project"></textarea>
            </div>
            <button className="btn btn-primary" type="button">Send</button>
          </form>
        </div>
      </div>
    </section>
  )
}
