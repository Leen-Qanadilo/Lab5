# pipeline/pipeline_job.py

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.dsl import pipeline


# ---------- WORKSPACE / COMPUTE SETTINGS ----------
SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106541"
WORKSPACE_NAME = "tumor_data_60106541"
COMPUTE_NAME = "brainTumor60106541"  # your existing compute cluster


# ---------- CONNECT TO WORKSPACE ----------
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)


# ---------- LOAD COMPONENTS ----------
# Phase 2: feature extraction component (your existing one)
extract_features_component = load_component("../components/component.yaml")

# Phase 4: baseline + GA feature selection component
feature_selection_ga_component = load_component(
    "../components/feature_selection_component.yaml"
)


# ---------- DEFINE PIPELINE ----------
@pipeline(default_compute=COMPUTE_NAME)
def tumor_pipeline(tumor_images_raw: Input):
    """
    Pipeline:
      1) extract_features_component: from raw images → features parquet
      2) feature_selection_ga_component: from features parquet → train/test splits + metrics + selected features
    """

    # Step 1: Feature extraction (Phase 2)
    feat_job = extract_features_component(
        input_data=tumor_images_raw
    )
    # feat_job.outputs.features_output : Parquet file from Phase 2

    # Step 2: Baseline + GA feature selection (Phase 4)
    fs_job = feature_selection_ga_component(
        features_parquet=feat_job.outputs.features_output
    )
    # fs_job.outputs.train_output         : train.parquet
    # fs_job.outputs.test_output          : test.parquet
    # fs_job.outputs.baseline_metrics     : baseline_metrics.json
    # fs_job.outputs.ga_metrics           : ga_metrics.json
    # fs_job.outputs.selected_features    : selected_features.json

    # Expose some outputs at the pipeline level (optional but useful)
    return {
        "features_parquet": feat_job.outputs.features_output,
        "train_parquet": fs_job.outputs.train_output,
        "test_parquet": fs_job.outputs.test_output,
        "baseline_metrics": fs_job.outputs.baseline_metrics,
        "ga_metrics": fs_job.outputs.ga_metrics,
        "selected_features": fs_job.outputs.selected_features,
    }


# ---------- BUILD & SUBMIT PIPELINE JOB ----------
if __name__ == "__main__":
   
    tumor_images_input = Input(
        type="uri_folder",
        path="azureml:tumor_data60106541:1",  # <-- change name/version if different
    )

    # Build pipeline job
    pipeline_job = tumor_pipeline(
        tumor_images_raw=tumor_images_input
    )

    pipeline_job.experiment_name = "lab5_phase4_pipeline"

    # Submit to Azure ML
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    print("Pipeline submitted!")
    print("Studio URL:", returned_job.studio_url)
