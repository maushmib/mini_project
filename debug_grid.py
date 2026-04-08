import csv, math, pandas as pd, string, numpy as np

# Read gps_log.csv
gps = []
with open('uploads/gps_log.csv') as f:
    for row in csv.DictReader(f):
        gps.append({'lat': float(row['lat']), 'lon': float(row['lon'])})

glats = [g['lat'] for g in gps]
glons = [g['lon'] for g in gps]
avg_lat = sum(glats)/len(glats)
avg_lon = sum(glons)/len(glons)

print(f"GPS rows      : {len(gps)}")
print(f"GPS lat range : {min(glats):.7f} to {max(glats):.7f}  spread={(max(glats)-min(glats))*111000:.3f}m")
print(f"GPS lon range : {min(glons):.7f} to {max(glons):.7f}  spread={(max(glons)-min(glons))*111000*math.cos(math.radians(avg_lat)):.3f}m")
print(f"GPS centroid  : {avg_lat:.7f}, {avg_lon:.7f}")

# CMD bounds from uploaded log
df = pd.read_csv('uploads/gps.csv', on_bad_lines='skip', engine='python', dtype=str)
MSG = df.columns[2]
cmd = df[df[MSG]=='CMD'].copy()
cmd['type'] = pd.to_numeric(cmd.iloc[:,6], errors='coerce')
cmd['lat']  = pd.to_numeric(cmd.iloc[:,11], errors='coerce')
cmd['lng']  = pd.to_numeric(cmd.iloc[:,12], errors='coerce')
nav = cmd[(cmd['type']==16) & cmd['lat'].notna() & (cmd['lat']!=0)]

print(f"\nCMD waypoints : {len(nav)}")

if nav.empty:
    print("NO CMD waypoints found in uploaded log!")
else:
    gml,gxl,gmn,gxn = nav.lat.min(),nav.lat.max(),nav.lng.min(),nav.lng.max()
    avg_cmd = (gml+gxl)/2
    cell_m = 2
    cl = cell_m/111000
    cn = cell_m/(111000*math.cos(math.radians(avg_cmd)))
    nr = max(1,math.ceil((gxl-gml)/cl))
    nc = max(1,math.ceil((gxn-gmn)/cn))

    print(f"CMD lat       : {gml:.7f} to {gxl:.7f}  spread={(gxl-gml)*111000:.1f}m")
    print(f"CMD lon       : {gmn:.7f} to {gxn:.7f}  spread={(gxn-gmn)*111000*math.cos(math.radians(avg_cmd)):.1f}m")
    print(f"CMD grid      : {nr} rows x {nc} cols = {nr*nc} cells")

    r = max(0, min(int((avg_lat - gml)/cl), nr-1))
    c = max(0, min(int((avg_lon - gmn)/cn), nc-1))
    L = list(string.ascii_uppercase)
    print(f"\nGPS inside CMD bbox?  lat:{gml<=avg_lat<=gxl}  lon:{gmn<=avg_lon<=gxn}")
    print(f"GPS centroid maps to: {L[c]}{r+1}  (row={r}, col={c})")
    print(f"\nAll unique GPS cells:")
    seen = set()
    for lat,lon in zip(glats,glons):
        rr = max(0,min(int((lat-gml)/cl),nr-1))
        cc = max(0,min(int((lon-gmn)/cn),nc-1))
        seen.add((rr,cc,L[cc]+str(rr+1)))
    for rr,cc,lbl in sorted(seen):
        print(f"  {lbl}  row={rr} col={cc}")
