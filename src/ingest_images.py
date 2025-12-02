# src/ingest_images.py

import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


ACCOUNT_NAME = "braintumor60106541"
CONTAINER_NAME = "raw"
LOCAL_DATASET = r"C:\Users\leenr\OneDrive\Desktop\brain_tumor_dataset"

print("Files in yes folder:", os.listdir(os.path.join(LOCAL_DATASET, "yes")))
print("Files in no folder:", os.listdir(os.path.join(LOCAL_DATASET, "no")))

def upload_folder(local_path, remote_path, container_client):
    for root, dirs, files in os.walk(local_path):
        for file in files:
            if file.startswith('.'):
                continue
            local_file = os.path.join(root, file)
            rel_path = os.path.relpath(local_file, local_path).replace("\\", "/")
            blob_path = f"{remote_path}/{rel_path}"
            print(f"Uploading {local_file} → {blob_path}")
            with open(local_file, "rb") as data:
                container_client.upload_blob(name=blob_path, data=data, overwrite=True)

def main():
    print("Authenticating with Azure...")
    credential = DefaultAzureCredential()
    blob_service = BlobServiceClient(
        account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net",
        credential=credential
    )

    # Create container if it does not exist
    try:
        container_client = blob_service.create_container(CONTAINER_NAME)
        print(f"Created container '{CONTAINER_NAME}'")
    except Exception:
        container_client = blob_service.get_container_client(CONTAINER_NAME)
        print(f"Using existing container '{CONTAINER_NAME}'")

    # Upload yes/ and no/
    upload_folder(os.path.join(LOCAL_DATASET, "yes"), "tumor_images/yes", container_client)
    upload_folder(os.path.join(LOCAL_DATASET, "no"), "tumor_images/no", container_client)

    print("Upload completed successfully!")

if __name__ == "__main__":
    main()
