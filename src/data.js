// src/data.js
export const profile = {
  name: "Joseph Paintsil",
  blurb:
    "I’m a geospatial data scientist who works across remote sensing, GIS, and machine learning. I also design and develop websites and interactive geospatial dashboards that make data easy to explore and understand. If you share these interests, connect with me on LinkedIn below.",
  location: "Lisbon, Portugal",
  email: "paintsil610@gmail.com",
  photo: "joseph2.png",
  years: "5+",
  subhead: "Exploring data, maps, and technology to make complex information easier to see and use.",
  socials: [
    { label: "GitHub", href: "https://github.com/JoeyPaintsil", icon: "Github" },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/joseph-paintsil/", icon: "Linkedin" },
  ],
};




export const stats = [
  { value: "4+", label: "Years of Experience" },
  { value: "15+", label: "Machine Learning Models Built" },
  { value: "50+", label: "Satellite Datasets Processed" },
  { value: "20+", label: "Tools and Frameworks" },
];


export const skills = [
  "Google Earth Engine", "Python", "NumPy", "Pandas", "Rasterio",
  "Geopandas", "xarray", "Scikit-learn", "SHAP", "GDAL",
  "ArcGIS Pro", "QGIS", "EO data: Sentinel, Landsat, NICFI Planet",
  "Cloud: GEE, GCP, AWS", "APIs", "Docker", "Git", "React"
]
export const projects = [
  {
    title: "Flood Risk Prediction Dashboard (Latest Project)!",
    year: "2025",
    tags: ["EO", "ML", "Flood Risk", "Web GIS"],
    description:
      "An interactive machine learning dashboard for predicting and visualizing flood risk anywhere in the world! .",
    links: [{ label: "View Project", href: "https://josephpaintsil.com/flood-mapper" }]
  },
  {
    title: "JoeLocator",
    year: "2024",
    tags: ["Web GIS", "Leaflet", "JavaScript"],
    description:
      "Search amenities within a chosen radius and download location data for analysis.",
    links: [
      { label: "View Project", href: "https://joeypaintsil.github.io/JoeLocator/" }
    ]
  },
  {
    title: "Restaurant Finder",
    year: "2024",
    tags: ["Python", "Database", "Leaflet", "Web"],
    description:
      "Campus-focused app to find nearby restaurants with navigation and filtering.",
    links: [
      { label: "View Project", href: "https://github.com/geotech-programming-project/restaurante-finder-final" }
    ]
  },
  {
    title: "Gmap Coordinate Converter",
    year: "2022",
    tags: ["Python", "GIS", "Automation"],
    description:
      "An app that Converts local Ghanaian coordinates to WGS84 and plots to Google Earth automatically.",
    links: [
      { label: "View Project", href: "https://drive.google.com/file/d/11XrPRNGos0KXFby_9gaWHoK-4ZKlrzqw/view" }
    ]
  },
  {
    title: "Austria University Site Suitability (Story Map)",
    year: "2024",
    tags: ["ArcGIS Pro", "Suitability", "Story Map"],
    description:
      "Multi-criteria site suitability analysis visualized as an interactive story map.",
    links: [
      { label: "View Project", href: "https://storymaps.arcgis.com/stories/c811309d16724735ad314468a91b343a" }
    ]
  },

  {
    title: "Air Quality Index (AQI) — Italy",
    year: "2023",
    tags: ["AQI", "Interpolation", "ArcGIS Pro"],
    description:
      "Computed AQI from NO₂, PM₂.₅ and PM₁₀ measurements and produced interpolated surfaces across Italy.",
    links: []
  }
];


export const experience = [
  {
    org: "Abeya",
    role: "Remote Sensing Data Scientist",
    period: "April 2025 – Present",
    bullets: [
      "Processing and analyzing satellite imagery for environmental and spatial insights.",
      "Developing reproducible workflows in Earth Observation and Machine Learning.",
      "Collaborating across teams to deliver accurate and timely geospatial products."
    ]
  },
  {
    org: "Licensed Professional Surveyor (Mr. Jonas Aryan Paintsil)",
    role: "Geomatic Engineer",
    period: "September 2022 – August 2023",
    bullets: [
      "Processed satellite imagery and produced thematic and topographic maps using ArcGIS.",
      "Conducted cadastral and engineering surveys ensuring spatial accuracy and data quality.",
      "Performed GNSS data collection and analysis for geospatial accuracy improvement.",
      "Used Python and Excel for survey computations and workflow automation."
    ]
  },
  {
    org: "Lands Commission",
    role: "Geomatic Engineer",
    period: "October 2021 – September 2022",
    bullets: [
      "Designed Python applications to automate coordinate conversion and mapping tasks.",
      "Processed remote sensing imagery for vegetation and land-use analyses.",
      "Supported data management and spatial analysis for cadastral and mapping projects.",
      "Contributed to digital transformation and process efficiency initiatives."
    ]
  },
  {
    org: "Licensed Professional Surveyor (Mr. Jonas Aryan Paintsil)",
    role: "Intern",
    period: "May 2020 – December 2020",
    bullets: [
      "Produced high-resolution orthophoto maps using LiDAR and drone data.",
      "Assisted in data acquisition, field surveying, and map production for infrastructure projects.",
      "Applied AutoCAD and ArcGIS for spatial data analysis and visualization."
    ]
  }
];

export const services = [
  {
    title: "Web GIS Dashboards",
    desc: "Interactive spatial dashboards and map interfaces built with React and MapLibre.",
    icon: "Map"
  },
  {
    title: "Web Development",
    desc: "Modern, responsive web apps using React, Tailwind CSS, and JavaScript frameworks.",
    icon: "Server"
  },
  {
    title: "EO & ML Pipelines",
    desc: "Python and Earth Engine workflows for analysis, modelling, and prediction.",
    icon: "Cpu"
  },
  {
    title: "Remote Sensing",
    desc: "Processing and visualization of satellite imagery for mapping and analytics.",
    icon: "BarChart3"
  }
];


// src/data.js
export const skillCategories = [
  {
    name: "GIS & Surveying",
    items: [
      { name: "Google Earth Engine", icon: "Globe2" },
      { name: "ArcGIS Pro", icon: "Layers" },
      { name: "QGIS", icon: "Map" },
      { name: "PostGIS / PostgreSQL", icon: "Database" },
      { name: "Leaflet / MapLibre", icon: "MapPinned" },
      { name: "AutoCAD", icon: "Ruler" },
      { name: "GNSS Processing (Topcon Tools)", icon: "Navigation" }
    ]
  },
  {
    name: "Web Development",
    items: [
      { name: "React", icon: "Cpu" },
      { name: "Tailwind CSS", icon: "Brush" },
      { name: "JavaScript", icon: "Code2" },
      { name: "HTML / CSS", icon: "FileCode" },
      { name: "Node.js", icon: "Network" },
      { name: "Flask / Express.js", icon: "Server" },
      { name: "FastAPI", icon: "ServerCog" },
      { name: "Docker", icon: "Boxes" },
      { name: "GCP", icon: "Cloud" },
      { name: "Git & GitHub", icon: "GitBranch" },
      { name: "VS Code / PyCharm", icon: "Laptop" }
    ]
  },
  {
    name: "EO & Machine Learning",
    items: [
      { name: "Sentinel-1 / 2", icon: "Satellite" },
      { name: "Landsat", icon: "Image" },
      { name: "Scikit-learn", icon: "BrainCircuit" },
      { name: "XGBoost", icon: "BarChart3" },
      { name: "Random Forest", icon: "TreePine" },
      { name: "Rasterio", icon: "Image" },
      { name: "xarray", icon: "Box" },
      { name: "Remote Sensing Analysis", icon: "Radar" }
    ]
  },
  {
    name: "Python & Data Science",
    items: [
      { name: "Python", icon: "Code" },
      { name: "Pandas", icon: "Table" },
      { name: "NumPy", icon: "FunctionSquare" },
      { name: "GeoPandas", icon: "MapPin" },
      { name: "Matplotlib", icon: "BarChartBig" },
      { name: "Plotly", icon: "ChartSpline" },
      { name: "SQL", icon: "Database" },
      { name: "ETL & Automation Scripts", icon: "Cog" },
      { name: "Jupyter Notebook", icon: "Notebook" },
      { name: "Excel", icon: "Sheet" }
    ]
  }
];



export const certifications = [
  "Esri Training: Spatial Analysis",
  "Google Cloud Fundamentals",
  "Remote Sensing for Land Cover"
];

export const currentlyLearning = ["PyTorch Geo", "Snowflake + Geo", "STAC APIs"];
