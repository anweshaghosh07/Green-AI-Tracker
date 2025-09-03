import mlflow
import time

mlflow.set_experiment("smoke_experiment")
with mlflow.start_run():
    mlflow.log_param("example_param", 42)
    mlflow.log_metric("example_metric", 3.14)
    time.sleep(0.5)
    mlflow.log_metric("example_metric", 4.56, step=1)
print("Logged a demo MLflow run.")
