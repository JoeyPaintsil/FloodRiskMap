
import { useEffect } from "react";

// src/App.jsx
import { Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import About from "./components/About";
import Services from "./components/Services";
import Skills from "./components/Skills";
import Projects from "./components/Projects";
import Experience from "./components/Experience";
import Contact from "./components/Contact";
import Footer from "./components/Footer";
import FloodMapper from "./pages/FloodMapper";
import FloodDashboard from "./pages/FloodDashboard";

export default function App() {
  const location = useLocation();
  useEffect(() => {
  if (location.hash === "#contact") {
    // Wait for components to load, then scroll
    setTimeout(() => {
      const contactSection = document.getElementById("contact");
      if (contactSection) {
        contactSection.scrollIntoView({ behavior: "smooth" });
      }
    }, 200);
  }
}, [location]);


  // Pages where Navbar/Footer should be hidden
  const isDashboard = location.pathname.startsWith("/flood-mapper");

  return (
    <div id="top">
      {!isDashboard && <Navbar />}

      <Routes>
        <Route
          path="/"
          element={
            <main>
              <Hero />
              <About />
              <Services />
              <Skills />
              <Projects />
              <Experience />
              <Contact />
            </main>
          }
        />
        <Route path="/flood-mapper" element={<FloodMapper />} />
        <Route path="/flood-mapper/dashboard" element={<FloodDashboard />} />
      </Routes>

      {!isDashboard && <Footer />}
    </div>
  );
}
