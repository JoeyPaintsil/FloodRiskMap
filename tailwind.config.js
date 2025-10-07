/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a"
        },
        brandOrange: {
          50:  "#fff7f0",
          100: "#ffe6d1",
          200: "#ffc299",
          300: "#ff9d61",
          400: "#ff7a1a",
          500: "#f56a00",
          600: "#e65e00",
          700: "#cc5200",
          800: "#994000",
          900: "#663000"
        }
      },
      boxShadow: {
        soft: "0 10px 30px rgba(2,6,23,.08)"
      }
    },
  },
  plugins: [],
}
