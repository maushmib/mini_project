import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {

  const [video, setVideo] = useState(null);
  const [csv, setCsv] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {

    if (!video || !csv) {
      alert("Select both files");
      return;
    }

    const formData = new FormData();
    formData.append("video", video);
    formData.append("csv", csv);

    setLoading(true);
    setResult(null);

    try {
      const res = await axios.post(
        "http://localhost:5000/process",
        formData,
        { responseType: "blob" }
      );

      setResult(URL.createObjectURL(res.data));
    } catch {
      alert("Processing failed");
    }

    setLoading(false);
  };


  return (
    <div className="app">

      <header className="header">
        Paddy Field Anomaly Detection System
      </header>

      <div className="container">

        {/* LEFT PANEL */}
        <div className="card input-card">

          <h2>Upload Data</h2>

          <label className="input-label">
            Drone Video
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setVideo(e.target.files[0])}
            />
          </label>

          <label className="input-label">
            GPS CSV Log
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setCsv(e.target.files[0])}
            />
          </label>

          <button
            className="btn"
            onClick={handleUpload}
            disabled={loading}
          >
            {loading ? "Processing..." : "Run Detection"}
          </button>

        </div>


        {/* RIGHT PANEL */}
        <div className="card output-card">

          <h2>Output Grid</h2>

          {!result && (
            <div className="placeholder">
              Result will appear here
            </div>
          )}

          {result && (
            <img src={result} alt="grid" />
          )}

        </div>

      </div>

    </div>
  );
}
