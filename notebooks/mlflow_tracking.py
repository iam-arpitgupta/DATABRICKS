import os 
import json 

import mlflow 
from dotenv import load_dotenv

#set up the mlflow for databrics 
def is_databricks() -> bool:
    """Check if the code is running in databricks """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


mlflow.set_tracking_uri()

if not is_databricks:
    load_dotenv()
    profile = os.environ.get("PROFILE")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


experiment = mlflow.set_experiment(experiment_name= "marvel_charcter")
mlflow.set_experiment_tags({"repository_name": "marvelousmlops/marvel-characters"})
print(experiment)
# dump artifacts into json for json visualization 
os.makedirs("../demo_artifacts", exist_ok=True)
with open("../demo_artifacts/mlflow_experiment.json", "w") as json_file:
    json.dump(experiment.__dict__, json_file, indent=4)


# we can even get the mlflow tracking using experiment id 
mlflow.get_experiment(experiment.experiment_id)
# search for experiment
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='marvelousmlops/marvel-characters'"
)
print(experiments)


# start a run 
mlflow.start_run()

# getting an specific run 
with mlflow.start_run(
    run_name="marvel-demo-run",
    tags={"git_sha": "1234567890abcd"},
    description="marvel character prediction demo run",
) as run:
    run_id = run.info.run_id
    mlflow.log_params({"type": "marvel_demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})
print(mlflow.active_run() is None)


run_info = mlflow.get_run(run_id=run_id).to_dictionary


# saving the run to the artifacts 
with open(".demo./artifacts/run_info.json","w") as json_file:
    json.dump(run_info , json_file , indent = 4)


print(run_info["data"]["metrics"])
print(run_info["data"]["params"])


run_id = mlflow.search_runs(
    experiment_name = "/Shared/marvel-demo",
    filter_string="tags.git_sha='1234567890abcd'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)






