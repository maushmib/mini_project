import os
import csv
import subprocess
import re

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from final_log import parse_log, build_map

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store for last run stats
_last_stats = {}


def write_gps_csv(df, path):
    """Write pos_df or gps_df to the CSV format expected by anomaly.py."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "lat", "lon", "alt", "speed"])
        for _, row in df.iterrows():
            ts = row["ts"]
            time_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if hasattr(ts, "strftime") else str(ts)
            lon = row.get("lng", row.get("lon", 0))   # pos_df uses 'lng', gps_df uses 'lng' too
            writer.writerow([time_str, row["lat"], lon, row["alt"], row.get("spd", 0)])


def parse_anomaly_coords(output_text):
    anomalies = []
    lines = output_text.splitlines()
    i = 0
    while i < len(lines):
        if "Hybrid plant detected at:" in lines[i]:
            lat, lng = None, None
            for j in range(i + 1, min(i + 5, len(lines))):
                m = re.search(r"Latitude:\s*([\d.\-]+)", lines[j])
                if m:
                    lat = float(m.group(1))
                m = re.search(r"Longitude:\s*([\d.\-]+)", lines[j])
                if m:
                    lng = float(m.group(1))
            if lat is not None and lng is not None:
                anomalies.append({"lat": lat, "lng": lng, "severity": "medium", "detections": 1})
        i += 1
    return anomalies


def parse_video_stats(output_text):
    """Extract FPS, total frames extracted, and total frames processed from anomaly.py stdout."""
    stats = {"fps": None, "frames_extracted": None, "frames_processed": None}
    for line in output_text.splitlines():
        m = re.search(r"FPS\s*:\s*([\d.]+)", line)
        if m:
            stats["fps"] = round(float(m.group(1)), 2)
        m = re.search(r"Total frames extracted\s*:\s*(\d+)", line)
        if m:
            stats["frames_extracted"] = int(m.group(1))
        m = re.search(r"Total frames processed\s*:\s*(\d+)", line)
        if m:
            stats["frames_processed"] = int(m.group(1))
    return stats


@app.route("/process", methods=["POST"])
def process_files():
    global _last_stats

    video    = request.files["video"]
    csv_file = request.files["csv"]

    video_path = os.path.join(UPLOAD_FOLDER, "video.mp4")
    log_csv    = os.path.join(UPLOAD_FOLDER, "gps.csv")
    gps_out    = os.path.join(UPLOAD_FOLDER, "gps_log.csv")
    grid_path  = os.path.join(UPLOAD_FOLDER, "grid.png")
    map_path   = os.path.join(UPLOAD_FOLDER, "map.html")

    video.save(video_path)
    csv_file.save(log_csv)

    # 1. Parse ArduPilot log
    try:
        pos_df, gps_df, last_mission = parse_log(log_csv)
    except Exception as e:
        return jsonify({"error": f"Log parsing failed: {e}"}), 400

    if pos_df.empty:
        return jsonify({"error": "No POS data found in log"}), 400

    # 2. Write GPS CSV for anomaly.py — use POS (high density) for better interpolation
    src_df = pos_df if not pos_df.empty else gps_df
    write_gps_csv(src_df, gps_out)
    print(f"[server] gps_log.csv written from {'POS' if not pos_df.empty else 'GPS'} "
          f"— {len(src_df)} rows → {gps_out}")

    # 3. Run anomaly detection — pass CMD bounding box so grid uses field area
    cmd_bounds = ""
    if not last_mission.empty:
        cmd_bounds = (
            f"{last_mission.lat.min()},{last_mission.lat.max()},"
            f"{last_mission.lng.min()},{last_mission.lng.max()}"
        )

    result = subprocess.run(
        ["python", "anomaly.py", video_path, gps_out, grid_path, cmd_bounds],
        capture_output=True, text=True
    )
    stdout = result.stdout + result.stderr

    # 4. Parse anomaly coords and video stats from stdout
    anomalies   = parse_anomaly_coords(stdout)
    video_stats = parse_video_stats(stdout)

    # 5. Build map — returns (path, cell_summary)
    _, cell_summary = build_map(pos_df, gps_df, last_mission, anomalies, map_path)

    # 6. Store stats for /stats endpoint
    flight_duration = (pos_df.ts.iloc[-1] - pos_df.ts.iloc[0]).total_seconds()
    _last_stats = {
        "anomaly_count":    len(anomalies),
        "pos_points":       len(pos_df),
        "gps_fixes":        len(gps_df),
        "mission_wps":      len(last_mission),
        "flight_duration":  int(flight_duration),
        "alt_mean":         round(float(pos_df.alt.mean()), 1),
        "alt_max":          round(float(pos_df.alt.max()), 1),
        "fps":              video_stats["fps"],
        "frames_extracted": video_stats["frames_extracted"],
        "frames_processed": video_stats["frames_processed"],
        "grid":             cell_summary,
    }

    return send_file(map_path, mimetype="text/html")


@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify(_last_stats)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
