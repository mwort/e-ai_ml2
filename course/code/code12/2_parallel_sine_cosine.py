import time
import math
import mlflow
from multiprocessing import Process

def run_sine(phase, name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Parallel Sine Comparison")

    with mlflow.start_run(run_name=name):
        mlflow.log_param("phase", phase)
        for step in range(240):
            t = step * 0.1
            value = math.sin(t + phase)*math.cos((t+phase)/10)
            mlflow.log_metric("sine", value, step=step)
            time.sleep(1)

if __name__ == "__main__":
    processes = [
        Process(target=run_sine, args=(0.0, "phase_0")),
        Process(target=run_sine, args=(math.pi/4, "phase_pi_over_4")),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

