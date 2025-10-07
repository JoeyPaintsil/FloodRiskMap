export default function Footer(){
  return (
    <footer className="py-10 border-t border-slate-200">
      <div className="container-xl text-sm text-slate-600 flex items-center justify-between">
        <p>© <span id="year"></span> Joseph Paintsil. All rights reserved.</p>

        <a className="hover:text-brand-700" href="#top">Back to top</a>
      </div>
      <script>{`document.getElementById('year').textContent = new Date().getFullYear();`}</script>
    </footer>
  )
}
