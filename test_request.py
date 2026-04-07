import requests

url = "http://127.0.0.1:5000/process"

files = {
    "video": open("wp1.mp4", "rb"),
    "csv": open("gps_log.csv", "rb")
}

response = requests.post(url, files=files)

with open("result.png", "wb") as f:
    f.write(response.content)

print("Result image saved as result.png")