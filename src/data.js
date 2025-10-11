// src/data.js
export const profile = {
  name: "Joseph Paintsil",
  blurb:
    "I’m a geospatial data scientist who works across remote sensing, GIS, and machine learning. I also design and develop websites and interactive geospatial dashboards that make data easy to explore and understand. If you share these interests, connect with me on LinkedIn below.",
  location: "Lisbon, Portugal",
  email: "paintsil610@gmail.com",
  photo: "./public/joseph2.png",
  years: "5+",
  subhead: "Exploring data, maps, and technology to make complex information easier to see and use.",
  socials: [
    { label: "GitHub", href: "https://github.com/JoeyPaintsil", icon: "Github" },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/joseph-paintsil/", icon: "Linkedin" },
  ],
};




export const stats = [
  { value: "25+", label: "Projects Completed" },
  { value: "6", label: "Countries Mapped" },
  { value: "10TB+", label: "EO Data Processed" },
  { value: "8+", label: "Models Deployed" }
];

export const skills = [
  "Google Earth Engine", "Python", "NumPy", "Pandas", "Rasterio",
  "Geopandas", "xarray", "Scikit-learn", "SHAP", "GDAL",
  "ArcGIS Pro", "QGIS", "EO data: Sentinel, Landsat, NICFI Planet",
  "Cloud: GEE, GCP, AWS", "APIs", "Docker", "Git", "React"
]

export const projects = [
  {
    title: "Cocoa Agroforestry Classifier",
    year: "2025",
    tags: ["Earth Engine", "Random Forest", "Sentinel-2", "NICFI"],
    description: "End-to-end pipeline mapping cocoa vs non-cocoa and shade intensity with wall-to-wall predictions and uncertainty.",
    links: [
      { label: "Code", href: "#" },
      { label: "Demo", href: "#" }
    ]
  },
  {
    title: "Global Flood Risk Prediction",
    year: "2025",
    tags: ["EO", "ML", "Hydrology", "GEE"],
    description: "Model to predict flood risk globally using EO features and global training points.",
    links: [
      { label: "Paper", href: "#" },
      { label: "Slides", href: "#" }
    ]
  },
  {
    title: "Above-ground Biomass & CO2e",
    year: "2024",
    tags: ["GEDI", "Sentinel-1/2", "PALSAR", "Carbon Accounting"],
    description: "AGBD to carbon and CO2e reporting aligned with GHG protocol with uncertainty estimates.",
    links: [
      { label: "Report", href: "#" }
    ]
  }
]

export const experience = [
  {
    org: "Abeya",
    role: "Remote Sensing Data Scientist",
    period: "April 2024 - Present",
    bullets: [
      "Built EO pipelines in GEE for classification and biomass mapping",
      "Delivered dashboards and reproducible analysis for clients",
      "Optimized data flows and cloud exports to GCS/S3"
    ]
  },
  {
    org: "TechFides LLC",
    role: "Manager, Gov & Nonprofit Ops",
    period: "2024 — Present",
    bullets: [
      "Led process improvement and analytics initiatives",
      "Managed bids, RFPs, and research deliverables"
    ]
  }
]

export const services = [
  {
    title: "Web GIS & Dashboards",
    desc: "Operational web maps and dashboards with React, MapLibre/Leaflet, and cloud tiles.",
    icon: "Map"
  },
  {
    title: "EO Pipelines & ML",
    desc: "End-to-end Earth Engine and Python pipelines for classification, change, biomass.",
    icon: "Cpu"
  },
  {
    title: "APIs & Data Products",
    desc: "Clean, versioned spatial APIs and reproducible data products for teams.",
    icon: "Server"
  },
  {
    title: "Risk & Monitoring",
    desc: "Flood, deforestation, crops, and custom monitoring with alerting and reports.",
    icon: "BarChart3"
  }
]

// src/data.js
export const skillCategories = [
  {
    name: "Core GIS",
    items: [
      { name: "Google Earth Engine", icon: "Globe2" },
      { name: "QGIS", icon: "Map" },
      { name: "ArcGIS Pro", icon: "Layers" }
    ]
  },
  {
    name: "Python & Data",
    items: [
      { name: "Python", icon: "Code2" },
      { name: "NumPy", icon: "FunctionSquare" },
      { name: "Pandas", icon: "Table" },
      { name: "Rasterio", icon: "Image" },
      { name: "GeoPandas", icon: "MapPin" },
      { name: "xarray", icon: "Box" }
    ]
  },
  {
    name: "EO & ML",
    items: [
      { name: "Sentinel-1/2", icon: "Satellite" },
      { name: "Landsat", icon: "Image" },
      { name: "NICFI Planet", icon: "Globe" },
      { name: "Scikit-learn", icon: "BrainCircuit" },
      { name: "SHAP", icon: "BarChart3" }
    ]
  },
  {
    name: "Cloud & Apps",
    items: [
      { name: "React", icon: "Cpu" },
      { name: "MapLibre/Leaflet", icon: "MapPinned" },
      { name: "GCP", icon: "Cloud" },
      { name: "AWS", icon: "Server" },
      { name: "Docker", icon: "Boxes" }
    ]
  }
];

export const certifications = [
  "Esri Training: Spatial Analysis",
  "Google Cloud Fundamentals",
  "Remote Sensing for Land Cover"
];

export const currentlyLearning = ["PyTorch Geo", "Snowflake + Geo", "STAC APIs"];
