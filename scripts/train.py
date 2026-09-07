import os,json,joblib,pandas as pd
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
D=Path("data"); M=Path("model_building"); M.mkdir(exist_ok=True); token=os.getenv("HF_TOKEN"); user=os.getenv("HF_USERNAME"); repo=os.getenv("HF_DATASET_REPO",f"{user}/tourism-wellness-dataset" if user else "")
trp=hf_hub_download(repo_id=repo,filename="train.csv",repo_type="dataset",token=token) if token and user else D/"train.csv"; tep=hf_hub_download(repo_id=repo,filename="test.csv",repo_type="dataset",token=token) if token and user else D/"test.csv"
tr=pd.read_csv(trp); te=pd.read_csv(tep); Xtr=tr.drop(columns="ProdTaken"); ytr=tr.ProdTaken; Xte=te.drop(columns="ProdTaken"); yte=te.ProdTaken
cat=Xtr.select_dtypes("object").columns.tolist(); num=[c for c in Xtr.columns if c not in cat]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
configs=[("Decision Tree",DecisionTreeClassifier(random_state=42,class_weight="balanced"),{"model__max_depth":[6,None],"model__min_samples_split":[5,10]}),("Random Forest",RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced"),{"model__n_estimators":[150,250],"model__max_depth":[10,20],"model__min_samples_leaf":[1,2]}),("Gradient Boosting",GradientBoostingClassifier(random_state=42),{"model__n_estimators":[100,200],"model__learning_rate":[.05,.1],"model__max_depth":[2,3]}),("AdaBoost",AdaBoostClassifier(random_state=42),{"model__n_estimators":[100,200],"model__learning_rate":[.5,1.]})]
rows=[]; best={}
for name,est,grid in configs:
 pipe=Pipeline([("preprocessor",pre),("model",est)]); gs=GridSearchCV(pipe,grid,scoring="roc_auc",cv=3,n_jobs=-1); gs.fit(Xtr,ytr); pred=gs.predict(Xte); prob=gs.predict_proba(Xte)[:,1]; rows.append({"Model":name,"Best Parameters":gs.best_params_,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,prob)}); best[name]=gs.best_estimator_
res=pd.DataFrame(rows).sort_values("roc_auc",ascending=False).reset_index(drop=True); res.to_csv(M/"experiment_results.csv",index=False); name=res.loc[0,"Model"]; joblib.dump(best[name],M/"model.joblib"); meta={"model_name":name,"target":"ProdTaken","selection_metric":"roc_auc","best_parameters":res.loc[0,"Best Parameters"],"metrics":{k:float(res.loc[0,k]) for k in ["accuracy","precision","recall","f1","roc_auc"]}}; (M/"model_metadata.json").write_text(json.dumps(meta,indent=2))
if token and user:
 mr=os.getenv("HF_MODEL_REPO",f"{user}/tourism-wellness-model"); api=HfApi(token=token); api.create_repo(repo_id=mr,repo_type="model",exist_ok=True,private=False); api.upload_folder(folder_path=str(M),repo_id=mr,repo_type="model",allow_patterns=["model.joblib","model_metadata.json","experiment_results.csv"])
print(res[["Model","accuracy","precision","recall","f1","roc_auc"]].round(4).to_string(index=False)); print("Selected:",name)
