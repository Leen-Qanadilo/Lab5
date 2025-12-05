from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command

# ---------- Workspace info ----------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106541"
WORKSPACE_NAME = "tumor_data_60106541"
COMPUTE_NAME = "brainTumor60106541"

# ---------- Connect to workspace ----------
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# ---------- Build the command job directly ----------
job = command(
    code="../src",  # folder that contains extract_features.py
    command=(
    "pip install scikit-image && "
    "python extract_features.py "
    "--input_data ${{inputs.input_data}} "
    "--output_path ${{outputs.features_output}}"
),
    inputs={
        "input_data": Input(
            type="uri_folder",
            path="azureml:tumor_data60106541:1", 
        )
    },
    outputs={
        "features_output": Output(
            type="uri_file",
            mode="rw_mount",
        )
    },
    environment="azureml:AzureML-sklearn-1.5:1",

    compute=COMPUTE_NAME,
    experiment_name="lab5_extract_features_test",
)

returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted!")
print("Studio URL:", returned_job.studio_url)
