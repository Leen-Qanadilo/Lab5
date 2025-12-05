from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, load_component

# 1) Connect to your workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="a485bb50-61aa-4b2f-bc7f-b6b53539b9d3",
    resource_group_name="rg-60106541",
    workspace_name="tumor_data_60106541",
)

# 2) Load the component from YAML
extract_features_component = load_component("components/component.yml")

# 3) Build the job
job = extract_features_component(
    input_data=Input(
        type="uri_folder",
        path="azureml:data:tumor_images_raw:1"  # adjust version if needed
    ),
    features_output=Output(
        type="uri_file",
        mode="rw_mount",
    ),
)

job.compute = "brainTumor60106541"
job.experiment_name = "lab5_extract_features_test"

# 4) Submit
returned_job = ml_client.jobs.create_or_update(job)
print(returned_job.studio_url)
