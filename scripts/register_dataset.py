import os
from pathlib import Path
from huggingface_hub import HfApi
token=os.getenv("HF_TOKEN"); user=os.getenv("HF_USERNAME")
if not token or not user: raise RuntimeError("Set HF_TOKEN and HF_USERNAME")
repo=os.getenv("HF_DATASET_REPO",f"{user}/tourism-wellness-dataset")
api=HfApi(token=token); api.create_repo(repo_id=repo,repo_type="dataset",exist_ok=True,private=False)
api.upload_file(path_or_fileobj="data/tourism.csv",path_in_repo="tourism.csv",repo_id=repo,repo_type="dataset")
print(f"Dataset registered: https://huggingface.co/datasets/{repo}")
