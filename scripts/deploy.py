import os
from huggingface_hub import HfApi
token=os.getenv("HF_TOKEN"); user=os.getenv("HF_USERNAME")
if not token or not user: raise RuntimeError("Set HF_TOKEN and HF_USERNAME")
repo=os.getenv("HF_SPACE_REPO",f"{user}/tourism-wellness-predictor"); api=HfApi(token=token); api.create_repo(repo_id=repo,repo_type="space",space_sdk="docker",exist_ok=True,private=False); api.upload_folder(folder_path="deployment",repo_id=repo,repo_type="space",path_in_repo="."); print(f"Space: https://huggingface.co/spaces/{repo}")
