import { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import TrainingPage from "./pages/TrainingPage";
import "./App.css";

function NavBar() {
  const location = useLocation();
  return (
    <nav style={{
      background: "linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)",
      padding: "0 2rem",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      height: "60px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.3)"
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <span style={{ fontSize: "24px" }}>🏛️</span>
        <div>
          <div style={{ color: "white", fontWeight: "700", fontSize: "16px", letterSpacing: "0.5px" }}>
            GST 2A Converter
          </div>
          <div style={{ color: "#90caf9", fontSize: "11px" }}>AI-Powered Statement Extractor</div>
        </div>
      </div>
      <div style={{ display: "flex", gap: "8px" }}>
        {[
          { path: "/", label: "📄 Upload & Extract" },
          { path: "/training", label: "🧠 Model Training" }
        ].map(({ path, label }) => (
          <Link
            key={path}
            to={path}
            style={{
              color: location.pathname === path ? "white" : "#90caf9",
              textDecoration: "none",
              padding: "8px 16px",
              borderRadius: "6px",
              background: location.pathname === path ? "rgba(255,255,255,0.15)" : "transparent",
              fontSize: "14px",
              fontWeight: location.pathname === path ? "600" : "400",
              transition: "all 0.2s"
            }}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <Router>
      <div style={{ minHeight: "100vh", background: "#f0f4f8", fontFamily: "Segoe UI, sans-serif" }}>
        <NavBar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </div>
    </Router>
  );
}