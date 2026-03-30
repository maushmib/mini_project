import os
import subprocess
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/process", methods=["POST"])
def process_files():

    video = request.files["video"]
    csv_file = request.files["csv"]

    video_path = os.path.join(UPLOAD_FOLDER, "video.mp4")
    csv_path = os.path.join(UPLOAD_FOLDER, "gps.csv")
    output_path = os.path.join(UPLOAD_FOLDER, "grid.png")

    video.save(video_path)
    csv_file.save(csv_path)

    # 🔥 RUN anomaly.py
    subprocess.run([
        "python",
        "anomaly.py",
        video_path,
        csv_path,
        output_path
    ])

    # 🔥 VERY IMPORTANT → return image file
    return send_file(
        output_path,
        mimetype="image/png"
    )


if __name__ == "__main__":
    app.run(debug=True)
