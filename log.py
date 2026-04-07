import csv

input_file = "final.csv"
output_file = "gps_log.csv"

with open(input_file, "r") as f, open(output_file, "w", newline="") as out:
    reader = csv.reader(f)
    writer = csv.writer(out)

    # header
    writer.writerow(["time","lat","lon","alt","speed"])

    for row in reader:

        if len(row) > 11 and row[2] == "GPS":
            try:
                time = row[1]
                lat = float(row[10])
                lon = float(row[11])
                alt = float(row[7])
                speed = float(row[9])

                writer.writerow([time,lat,lon,alt,speed])

            except:
                pass

print("GPS log created successfully")