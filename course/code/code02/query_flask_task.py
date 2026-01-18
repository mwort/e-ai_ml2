import sys, requests

tid = int(sys.argv[1])
r = requests.get(f"http://127.0.0.1:5000/tasks/{tid}")

print(r.status_code)
print(r.json())

