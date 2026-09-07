import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi,hf_hub_download
D=Path("data"); token=os.getenv("HF_TOKEN"); user=os.getenv("HF_USERNAME"); repo=os.getenv("HF_DATASET_REPO",f"{user}/tourism-wellness-dataset" if user else "")
source=hf_hub_download(repo_id=repo,filename="tourism.csv",repo_type="dataset",token=token) if token and user else D/"tourism.csv"
df=pd.read_csv(source).drop(columns=[c for c in ["Unnamed: 0","CustomerID"] if c in pd.read_csv(source,nrows=1).columns])
df["Gender"]=df["Gender"].replace({"Fe Male":"Female"}); df["MaritalStatus"]=df["MaritalStatus"].replace({"Unmarried":"Single"})
tr,te=train_test_split(df,test_size=.2,random_state=42,stratify=df.ProdTaken); tr.to_csv(D/"train.csv",index=False); te.to_csv(D/"test.csv",index=False)
if token and user:
 api=HfApi(token=token)
 for f in ["train.csv","test.csv"]: api.upload_file(path_or_fileobj=str(D/f),path_in_repo=f,repo_id=repo,repo_type="dataset")
print(tr.shape,te.shape)
