import { useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:5000";

const CELL_COLORS = { "#ffffff": "#e5e7eb", "#FFE000": "#FFE000", "#FF8C00": "#FF8C00", "#FF3333": "#FF3333" };

export default function App() {
  const [video, setVideo]   = useState(null);
  const [csv, setCsv]       = useState(null);
  const [mapUrl, setMapUrl] = useState(null);
  const [stats, setStats]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(null);

  const handleUpload = async () => {
    if (!video || !csv) { setError("Select both files"); return; }
    setError(null);
    setLoading(true);
    setMapUrl(null);
    setStats(null);

    const formData = new FormData();
    formData.append("video", video);
    formData.append("csv", csv);

    try {
      const res = await axios.post(`${API}/process`, formData, { responseType: "blob" });
      const blob = new Blob([res.data], { type: "text/html" });
      setMapUrl(URL.createObjectURL(blob));

      const s = await axios.get(`${API}/stats`);
      setStats(s.data);
    } catch (e) {
      setError("Processing failed. Check server logs.");
    }

    setLoading(false);
  };

  // Build 2x2 grid display from cell_summary
  const gridCells = stats?.grid ?? [];
  const cellMap = {};
  gridCells.forEach(c => { cellMap[c.cell] = c; });

  return (
    <div className="app">
      <header className="header">
        <span className="header-icon">🌾</span>
        Paddy Field Anomaly Detection System
      </header>

      <div className="container">

        {/* ── LEFT PANEL ── */}
        <div className="sidebar">

          {/* Upload */}
          <div className="card">
            <h2>Upload Data</h2>

            <label className="input-label">
              🎥 Drone Video
              <input type="file" accept="video/*" onChange={e => setVideo(e.target.files[0])} />
              {video && <span className="file-name">{video.name}</span>}
            </label>

            <label className="input-label">
              📄 ArduPilot Log CSV
              <input type="file" accept=".csv" onChange={e => setCsv(e.target.files[0])} />
              {csv && <span className="file-name">{csv.name}</span>}
            </label>

            {error && <div className="error">{error}</div>}

            <button className="btn" onClick={handleUpload} disabled={loading}>
              {loading ? <><span className="spinner" /> Processing…</> : "▶ Run Detection"}
            </button>
          </div>

          {/* Stats */}
          {stats && (
            <div className="card stats-card">
              <h2>Flight Summary</h2>
              <div className="stat-row"><span>Duration</span><b>{Math.floor(stats.flight_duration/60)}m {stats.flight_duration%60}s</b></div>
              <div className="stat-row"><span>FPS</span><b>{stats.fps ?? "—"}</b></div>
              <div className="stat-row"><span>Frames extracted</span><b>{stats.frames_extracted ?? "—"}</b></div>
              <div className="stat-row"><span>Frames processed</span><b>{stats.frames_processed ?? "—"}</b></div>
              <div className="stat-row"><span>POS points</span><b>{stats.pos_points}</b></div>
              <div className="stat-row"><span>GPS fixes</span><b>{stats.gps_fixes}</b></div>
              <div className="stat-row"><span>Mission WPs</span><b>{stats.mission_wps}</b></div>
              <div className="stat-row"><span>Alt mean / max</span><b>{stats.alt_mean} / {stats.alt_max} m</b></div>

              <div className="divider" />

              <h2>Anomaly Summary</h2>
              <div className="anomaly-badge">{stats.anomaly_count} detection{stats.anomaly_count !== 1 ? "s" : ""}</div>

              {/* 2m x 2m grid heatmap summary */}
              <h3 className="grid-title">2m × 2m Grid Cells</h3>
              <div className="grid-stats">
                <span>{gridCells.length} total cells</span>
                <span>{gridCells.filter(c => c.count > 0).length} with detections</span>
              </div>
              <div className="grid-table-wrap">
                <table className="grid-table">
                  <thead>
                    <tr><th>Cell</th><th>Detections</th><th>Level</th></tr>
                  </thead>
                  <tbody>
                    {gridCells.filter(c => c.count > 0).length === 0 ? (
                      <tr><td colSpan={3} style={{textAlign:"center",color:"#9ca3af"}}>No detections</td></tr>
                    ) : (
                      gridCells
                        .filter(c => c.count > 0)
                        .sort((a, b) => b.count - a.count)
                        .map(c => (
                          <tr key={c.cell}>
                            <td><b>{c.cell}</b></td>
                            <td>{c.count}</td>
                            <td><span className="level-dot" style={{background: CELL_COLORS[c.color] ?? c.color}} /></td>
                          </tr>
                        ))
                    )}
                  </tbody>
                </table>
              </div>
              <div className="heatmap-legend">
                <span style={{background:"#e5e7eb"}} /> None
                <span style={{background:"#FFE000"}} /> Low
                <span style={{background:"#FF8C00"}} /> Med
                <span style={{background:"#FF3333"}} /> High
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div className="card map-card">
          <h2>Flight Map &amp; Detections</h2>

          {loading && (
            <div className="map-loading">
              <span className="spinner large" />
              <p>Running CNN detection and building map…</p>
            </div>
          )}

          {!mapUrl && !loading && (
            <div className="placeholder">
              Upload files and click <b>Run Detection</b> to view the interactive map
            </div>
          )}

          {mapUrl && (
            <iframe
              src={mapUrl}
              title="Flight Map"
              className="map-frame"
            />
          )}
        </div>

      </div>
    </div>
  );
}
