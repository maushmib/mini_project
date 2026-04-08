import os
import sys

try:
    import pandas as pd
    import folium
    from folium.plugins import MeasureControl, MousePosition
except ImportError as exc:
    name = getattr(exc, "name", None) or str(exc).split()[-1]
    print(f"Missing dependency: {name}")
    raise


def safe_float(series):
    return pd.to_numeric(series, errors="coerce")


def parse_log(input_csv):
    """Parse ArduPilot CSV log. Returns (pos_df, gps_df, last_mission)."""
    df = pd.read_csv(input_csv, on_bad_lines="skip", engine="python", dtype=str)
    MSG = df.columns[2]

    # POS – continuous position
    pos_raw = df[df[MSG] == "POS"].copy()
    pos_raw["ts"]  = pd.to_datetime(pos_raw.iloc[:, 1], errors="coerce")
    pos_raw["lat"] = safe_float(pos_raw.iloc[:, 4])
    pos_raw["lng"] = safe_float(pos_raw.iloc[:, 5])
    pos_raw["alt"] = safe_float(pos_raw.iloc[:, 6])
    pos_df = (
        pos_raw[["ts", "lat", "lng", "alt"]]
        .dropna(subset=["ts", "lat", "lng"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    pos_df["lat"] = pos_df["lat"].rolling(window=10, center=True, min_periods=1).mean()
    pos_df["lng"] = pos_df["lng"].rolling(window=10, center=True, min_periods=1).mean()

    # GPS – raw fixes
    gps_raw = df[df[MSG] == "GPS"].copy()
    gps_raw["ts"]  = pd.to_datetime(gps_raw.iloc[:, 1], errors="coerce")
    gps_raw["lat"] = safe_float(gps_raw.iloc[:, 10])
    gps_raw["lng"] = safe_float(gps_raw.iloc[:, 11])
    gps_raw["alt"] = safe_float(gps_raw.iloc[:, 12])
    gps_raw["spd"] = safe_float(gps_raw.iloc[:, 13])
    gps_df = (
        gps_raw[["ts", "lat", "lng", "alt", "spd"]]
        .dropna(subset=["ts", "lat", "lng"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    gps_df["lat"] = gps_df["lat"].rolling(window=3, center=True, min_periods=1).mean()
    gps_df["lng"] = gps_df["lng"].rolling(window=3, center=True, min_periods=1).mean()

    # CMD – planned mission waypoints
    cmd_raw = df[df[MSG] == "CMD"].copy()
    cmd_raw["ts"]   = pd.to_datetime(cmd_raw.iloc[:, 1], errors="coerce")
    cmd_raw["seq"]  = safe_float(cmd_raw.iloc[:, 5])
    cmd_raw["type"] = safe_float(cmd_raw.iloc[:, 6])
    cmd_raw["lat"]  = safe_float(cmd_raw.iloc[:, 11])
    cmd_raw["lng"]  = safe_float(cmd_raw.iloc[:, 12])
    cmd_raw["alt"]  = safe_float(cmd_raw.iloc[:, 13])

    nav_wps = (
        cmd_raw[(cmd_raw["type"] == 16) & cmd_raw["lat"].notna() & (cmd_raw["lat"] != 0)]
        .sort_values(["ts", "seq"])
        .reset_index(drop=True)
    )

    if not nav_wps.empty:
        nav_wps["upload"] = nav_wps["ts"].diff().dt.total_seconds().gt(1).cumsum()
        missions = {u: g for u, g in nav_wps.groupby("upload")}
        last_mission = missions[max(missions.keys())].copy()
    else:
        last_mission = pd.DataFrame(columns=["lat", "lng", "alt", "seq"])

    return pos_df, gps_df, last_mission


def build_grid_heatmap(m, last_mission, anomalies):
    """
    Tile the CMD waypoint bounding box with 2m x 2m cells.
    Count anomaly detections per cell and render as a heatmap overlay.
    """
    import math
    import string

    if last_mission.empty:
        return []

    lats = last_mission.lat.dropna()
    lngs = last_mission.lng.dropna()
    min_lat, max_lat = lats.min(), lats.max()
    min_lng, max_lng = lngs.min(), lngs.max()

    avg_lat = (min_lat + max_lat) / 2

    # Convert 2 metres to degrees
    cell_m      = 2.0
    cell_dlat   = cell_m / 111000
    cell_dlng   = cell_m / (111000 * math.cos(math.radians(avg_lat)))

    n_rows = max(1, math.ceil((max_lat - min_lat) / cell_dlat))
    n_cols = max(1, math.ceil((max_lng - min_lng) / cell_dlng))

    # Count anomalies per (row, col)
    counts = [[0] * n_cols for _ in range(n_rows)]
    for a in anomalies:
        r = int((a["lat"] - min_lat) / cell_dlat)
        c = int((a["lng"] - min_lng) / cell_dlng)
        r = max(0, min(r, n_rows - 1))
        c = max(0, min(c, n_cols - 1))
        counts[r][c] += 1

    max_count = max(counts[r][c] for r in range(n_rows) for c in range(n_cols)) or 1

    def cell_color(count):
        if count == 0:
            return "#ffffff"
        ratio = count / max_count
        if ratio <= 0.33:
            return "#FFE000"
        elif ratio <= 0.66:
            return "#FF8C00"
        return "#FF3333"

    # Column labels: A, B, C … Z, AA, AB …
    def col_label(c):
        letters = string.ascii_uppercase
        if c < 26:
            return letters[c]
        return letters[c // 26 - 1] + letters[c % 26]

    grid_layer = folium.FeatureGroup(name="🟥 2m×2m Anomaly Heatmap", show=True)
    cell_summary = []

    for r in range(n_rows):
        for c in range(n_cols):
            lat_min = min_lat + r * cell_dlat
            lat_max = lat_min + cell_dlat
            lng_min = min_lng + c * cell_dlng
            lng_max = lng_min + cell_dlng

            count = counts[r][c]
            color = cell_color(count)
            label = f"{col_label(c)}{r + 1}"

            folium.Rectangle(
                bounds=[[lat_min, lng_min], [lat_max, lng_max]],
                color="#333333",
                weight=0.8,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                tooltip=f"Cell {label} — {count} detection(s)",
                popup=folium.Popup(
                    f"<b>Cell {label}</b><br>Anomalies: {count}<br>"
                    f"Lat: {lat_min:.6f} – {lat_max:.6f}<br>"
                    f"Lng: {lng_min:.6f} – {lng_max:.6f}",
                    max_width=220
                )
            ).add_to(grid_layer)

            if count > 0:
                div_html = (
                    '<div style="font-size:10px;font-weight:bold;color:#1a1a1a;'
                    'background:rgba(255,255,255,0.75);padding:1px 3px;'
                    'border-radius:3px;text-align:center;line-height:1.3;">'
                    + label + "<br>" + str(count) + "</div>"
                )
                folium.Marker(
                    location=[(lat_min + lat_max) / 2, (lng_min + lng_max) / 2],
                    icon=folium.DivIcon(html=div_html, icon_size=(36, 28), icon_anchor=(18, 14))
                ).add_to(grid_layer)

            cell_summary.append({"cell": label, "count": count, "color": color})

    grid_layer.add_to(m)
    print(f"Grid: {n_rows} rows x {n_cols} cols ({n_rows * n_cols} cells, 2m x 2m each)")
    return cell_summary


def build_map(pos_df, gps_df, last_mission, anomalies, output_html):
    """
    Build and save a Folium map.
    anomalies: list of dicts with keys lat, lng, severity, detections, description
    """
    all_lats = list(pos_df.lat) + list(last_mission.lat.dropna())
    all_lngs = list(pos_df.lng) + list(last_mission.lng.dropna())
    center_lat = (min(all_lats) + max(all_lats)) / 2
    center_lng = (min(all_lngs) + max(all_lngs)) / 2

    m = folium.Map(location=[center_lat, center_lng], zoom_start=19, tiles=None)

    folium.TileLayer(
        tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite", name="🛰 Satellite",
        subdomains=["mt0", "mt1", "mt2", "mt3"], max_zoom=22,
    ).add_to(m)
    folium.TileLayer(tiles="OpenStreetMap", name="🗺 OpenStreetMap").add_to(m)

    # POS path
    pos_layer = folium.FeatureGroup(name="🔴 Actual GPS Path (POS log)", show=True)
    folium.PolyLine(list(zip(pos_df.lat, pos_df.lng)), color="#FF3333", weight=3,
                    opacity=0.9, tooltip="Actual flight path from POS log").add_to(pos_layer)
    pos_layer.add_to(m)

    # GPS raw fixes
    gps_layer = folium.FeatureGroup(name="🟠 Raw GPS Fixes", show=False)
    folium.PolyLine(list(zip(gps_df.lat, gps_df.lng)), color="#FF8C00", weight=2,
                    opacity=0.7, dash_array="6 4", tooltip="Raw GPS fixes").add_to(gps_layer)
    gps_layer.add_to(m)

    # Planned mission (CMD)
    if not last_mission.empty:
        mission_layer = folium.FeatureGroup(name="🟡 Planned Mission Path (CMD)", show=True)
        folium.PolyLine(list(zip(last_mission.lat, last_mission.lng)), color="#FFE000",
                        weight=3, opacity=0.95, tooltip="Planned mission waypoints").add_to(mission_layer)
        for _, wp in last_mission.iterrows():
            folium.CircleMarker(
                location=[wp.lat, wp.lng], radius=5, color="#FFE000",
                fill=True, fill_color="#FFE000", fill_opacity=0.9,
                tooltip=f"WP {int(wp.seq)}  ({wp.lat:.7f}, {wp.lng:.7f})  Alt: {wp.alt:.1f}m",
            ).add_to(mission_layer)
        mission_layer.add_to(m)

    # Takeoff / landing
    home_lat, home_lng = pos_df.lat.iloc[0], pos_df.lng.iloc[0]
    end_lat,  end_lng  = pos_df.lat.iloc[-1], pos_df.lng.iloc[-1]
    start_ts = pos_df.ts.iloc[0].strftime("%H:%M:%S")
    end_ts   = pos_df.ts.iloc[-1].strftime("%H:%M:%S")

    markers_layer = folium.FeatureGroup(name="📍 Takeoff / Landing", show=True)
    folium.Marker([home_lat, home_lng],
                  popup=folium.Popup(f"<b>🛫 Takeoff</b><br>{home_lat:.7f}, {home_lng:.7f}<br>{start_ts}", max_width=250),
                  icon=folium.Icon(color="green", icon="plane", prefix="fa"),
                  tooltip="Takeoff / Home").add_to(markers_layer)
    folium.Marker([end_lat, end_lng],
                  popup=folium.Popup(f"<b>🛬 Landing</b><br>{end_lat:.7f}, {end_lng:.7f}<br>{end_ts}", max_width=250),
                  icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                  tooltip="Landing").add_to(markers_layer)
    markers_layer.add_to(m)

    # Anomaly detections
    if anomalies:
        anomaly_layer = folium.FeatureGroup(name="🟠 Anomaly Detections", show=True)
        for a in anomalies:
            folium.CircleMarker(
                location=[a["lat"], a["lng"]], radius=10,
                color="#FF8C00", fill=True, fill_color="#FF8C00", fill_opacity=0.85,
                tooltip=f"Anomaly | {a.get('severity','?')} | {a.get('detections',1)} detections",
                popup=folium.Popup(
                    f"<b>Anomaly</b><br>Lat: {a['lat']:.7f}<br>Lng: {a['lng']:.7f}<br>"
                    f"Severity: {a.get('severity','?')}<br>Detections: {a.get('detections',1)}", max_width=250)
            ).add_to(anomaly_layer)
            box = [[a["lat"] - 0.00002, a["lng"] - 0.000015],
                   [a["lat"] + 0.00002, a["lng"] + 0.000015]]
            folium.Rectangle(bounds=box, color="#FFFFFF", weight=2, fill=False,
                              tooltip="Anomaly bounding box").add_to(anomaly_layer)
        anomaly_layer.add_to(m)

    # 2x2 grid heatmap over CMD region
    cell_summary = build_grid_heatmap(m, last_mission, anomalies)

    # Legend
    anomaly_legend = "".join(
        f'<span style="color:#FF8C00;">●</span>&nbsp; Anomaly ({a.get("severity","?")})<br>'
        for a in anomalies
    ) if anomalies else '<span style="color:#aaa;">No anomalies detected</span><br>'

    legend_html = f"""
<div style="position:fixed;bottom:40px;left:40px;z-index:9999;
background-color:rgba(20,30,50,0.92);color:white;padding:14px 18px;border-radius:8px;
font-family:Arial,sans-serif;font-size:13px;border:1px solid rgba(255,255,255,0.2);
box-shadow:0 2px 12px rgba(0,0,0,0.5);min-width:200px;">
<b style="font-size:14px;">📡 Flight Path Legend</b><br><br>
<span style="color:#FF3333;">━━</span>&nbsp; Actual GPS Path (POS)<br>
<span style="color:#FF8C00;">╌╌</span>&nbsp; Raw GPS Fixes<br>
<span style="color:#FFE000;">━━</span>&nbsp; Planned Mission (CMD)<br><br>
<b style="font-size:13px;">🟥 2×2 Grid Heatmap</b><br><br>
<span style="background:#FFE000;padding:0 6px;">&nbsp;</span>&nbsp; Low anomalies<br>
<span style="background:#FF8C00;padding:0 6px;">&nbsp;</span>&nbsp; Medium anomalies<br>
<span style="background:#FF3333;padding:0 6px;">&nbsp;</span>&nbsp; High anomalies<br><br>
<b style="font-size:13px;">🌾 Anomaly Legend</b><br><br>
{anomaly_legend}
</div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # Flight stats
    flight_duration = (pos_df.ts.iloc[-1] - pos_df.ts.iloc[0]).total_seconds()
    stats_html = f"""
<div style="position:fixed;top:20px;right:20px;z-index:9999;
background-color:rgba(20,30,50,0.92);color:white;padding:14px 18px;border-radius:8px;
font-family:Arial,sans-serif;font-size:13px;border:1px solid rgba(255,255,255,0.2);
box-shadow:0 2px 12px rgba(0,0,0,0.5);min-width:210px;">
<b style="font-size:14px;">✈️ Flight Summary</b><br><br>
<b>Duration:</b> {int(flight_duration//60)}m {int(flight_duration%60)}s<br>
<b>Start:</b> {start_ts} &nbsp; <b>End:</b> {end_ts}<br><br>
<b>POS points:</b> {len(pos_df)}<br>
<b>GPS fixes:</b> {len(gps_df)}<br>
<b>Mission WPs:</b> {len(last_mission)}<br>
<b>Anomalies:</b> {len(anomalies)}<br><br>
<b>Alt (mean/max):</b> {pos_df.alt.mean():.1f} / {pos_df.alt.max():.1f} m<br>
<b>Home:</b> {home_lat:.7f}, {home_lng:.7f}<br>
</div>"""
    m.get_root().html.add_child(folium.Element(stats_html))

    MeasureControl(primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)
    MousePosition(position="bottomright", separator=" | Lng: ", prefix="Lat: ", num_digits=7).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    m.save(output_html)
    return output_html, cell_summary


# ── Standalone CLI usage ──────────────────────────────────────────────────────
if __name__ == "__main__":
    input_csv  = sys.argv[1] if len(sys.argv) > 1 else "TNAU2.csv"
    output_html = "drone_flight_map.html"

    print(f"Loading log: {input_csv}")
    pos_df, gps_df, last_mission = parse_log(input_csv)
    print(f"POS: {len(pos_df)}  GPS: {len(gps_df)}  Mission WPs: {len(last_mission)}")

    build_map(pos_df, gps_df, last_mission, anomalies=[], output_html=output_html)
    print(f"✅  Map saved → {output_html}")
