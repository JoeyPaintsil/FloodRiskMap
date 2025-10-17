// src/components/Contact.jsx
import { useState } from "react";

const SCRIPT_URL = import.meta.env.VITE_CONTACT_FORM_SCRIPT_URL;// <-- from Apps Script deployment

export default function Contact() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    subject: "General Inquiry",
    message: "",
  });
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState(null); // 'ok' | 'error' | null

  const onChange = (e) =>
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  async function onSubmit(e) {
    e.preventDefault();
    setSending(true);
    setStatus(null);

    try {
      // Send as x-www-form-urlencoded to avoid CORS preflight
      const body = new URLSearchParams(form);
      const res = await fetch(SCRIPT_URL, { method: "POST", body });

      let ok = res.ok;
      try {
        const json = await res.json();
        ok = ok && json.ok;
      } catch (_) {}

      if (ok) {
        setStatus("ok");
        setForm({ name: "", email: "", subject: "General Inquiry", message: "" });
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    } finally {
      setSending(false);
    }
  }

  return (
    <section id="contact" className="section bg-slate-50/70">
      <div className="container-xl grid md:grid-cols-3 gap-8 items-start">
        <div className="md:col-span-1">
          <h2 className="text-2xl font-bold">Contact</h2>
          {status === "ok" && (
            <p className="mt-3 text-emerald-600 text-sm">
              ✅ Thanks! Your message has been sent.
            </p>
          )}
          {status === "error" && (
            <p className="mt-3 text-rose-600 text-sm">
              ❌ Sorry—something went wrong. Please try again or email me directly.
            </p>
          )}
        </div>

        <div className="md:col-span-2 card p-6">
          <form className="grid gap-4" onSubmit={onSubmit}>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-slate-600">Name</label>
                <input
                  name="name"
                  type="text"
                  className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300"
                  placeholder="Your name"
                  value={form.name}
                  onChange={onChange}
                  required
                />
              </div>
              <div>
                <label className="text-sm text-slate-600">Email</label>
                <input
                  name="email"
                  type="email"
                  className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300"
                  placeholder="you@example.com"
                  value={form.email}
                  onChange={onChange}
                  required
                />
              </div>
            </div>

            {/* Subject dropdown */}
            <div>
              <label className="text-sm text-slate-600">Subject</label>
              <select
                name="subject"
                className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300 bg-white"
                value={form.subject}
                onChange={onChange}
              >
                <option>Flood Dashboard Issue</option>
                <option>General Inquiry</option>
                <option>Other</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-slate-600">Message</label>
              <textarea
                name="message"
                rows="5"
                className="mt-1 w-full border border-slate-300 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-300"
                placeholder=""
                value={form.message}
                onChange={onChange}
                required
              />
            </div>

            {/* Submit Button */}
            <button
                className="btn btn-primary flex justify-center items-center w-full"
                type="submit"
                disabled={sending}
              >
                {sending ? "Sending..." : "Send"}
              </button>


          </form>
        </div>
      </div>
    </section>
  );
}
