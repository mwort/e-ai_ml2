from flask import Flask, request, jsonify, abort

app = Flask(__name__)
tasks = {}
next_id = 1

@app.post("/tasks")  # Demo only
def create():
    global next_id
    if not request.json:
        abort(400)
    task = {
        "id": next_id,
        "state": "created",
        "data": request.json
    }
    tasks[next_id] = task
    next_id += 1
    return jsonify(task), 201

@app.get("/tasks")  # Demo only, would not use GET in real application
def list_tasks():
    return jsonify(list(tasks.values()))

@app.get("/tasks/<int:task_id>") # Demo only
def get_task(task_id):
    if task_id not in tasks:
        abort(404)
    return jsonify(tasks[task_id])

app.run()

