import mlflow

mlflow.set_experiment("Demo")

with mlflow.start_run():
    mlflow.log_param("lr", 1e-3)
    mlflow.log_metric("loss", 0.42)

