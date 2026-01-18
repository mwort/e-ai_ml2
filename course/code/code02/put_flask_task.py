import requests

r = requests.post(
  "http://127.0.0.1:5000/tasks",
  json={"type":"demo","params":{"x":1}} )
print(r.status_code)
print(r.json())
