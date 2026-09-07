import os
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download
st.set_page_config(page_title="Wellness Package Predictor",page_icon="✈️")
REPO=os.getenv("HF_MODEL_REPO","")
if not REPO: st.error("HF_MODEL_REPO is not configured."); st.stop()
@st.cache_resource
def load_model():
 p=hf_hub_download(repo_id=REPO,filename="model.joblib",repo_type="model"); return joblib.load(p)
model=load_model(); st.title("✈️ Wellness Tourism Package Predictor"); st.write("Estimate the probability that a customer will purchase the Wellness Tourism Package.")
col1,col2=st.columns(2)
with col1:
 age=st.number_input("Age",18,100,35); contact=st.selectbox("Type of Contact",["Self Enquiry","Company Invited"]); city=st.selectbox("City Tier",[1,2,3]); occupation=st.selectbox("Occupation",["Salaried","Small Business","Large Business","Free Lancer"]); gender=st.selectbox("Gender",["Male","Female"]); persons=st.number_input("Number of Persons Visiting",1,10,2); star=st.selectbox("Preferred Property Star",[3,4,5]); marital=st.selectbox("Marital Status",["Single","Married","Divorced"])
with col2:
 trips=st.number_input("Number of Trips",1.,30.,4.); passport=st.selectbox("Passport",[0,1],format_func=lambda x:"Yes" if x else "No"); car=st.selectbox("Own Car",[0,1],format_func=lambda x:"Yes" if x else "No"); children=st.number_input("Number of Children Visiting",0,5,0); designation=st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"]); income=st.number_input("Monthly Income",0.,200000.,25000.,1000.); pitch=st.selectbox("Pitch Satisfaction Score",[1,2,3,4,5]); product=st.selectbox("Product Pitched",["Basic","Deluxe","Standard","Super Deluxe","King"]); follow=st.number_input("Number of Follow-ups",1,10,2); duration=st.number_input("Duration of Pitch",1.,180.,15.)
if st.button("Predict Purchase",type="primary"):
 x=pd.DataFrame([{"Age":age,"TypeofContact":contact,"CityTier":city,"Occupation":occupation,"Gender":gender,"NumberOfPersonVisiting":persons,"PreferredPropertyStar":star,"MaritalStatus":marital,"NumberOfTrips":trips,"Passport":passport,"OwnCar":car,"NumberOfChildrenVisiting":children,"Designation":designation,"MonthlyIncome":income,"PitchSatisfactionScore":pitch,"ProductPitched":product,"NumberOfFollowups":follow,"DurationOfPitch":duration}]); prob=float(model.predict_proba(x)[0,1]); pred=int(model.predict(x)[0]); (st.success if pred else st.info)(f"{'Likely' if pred else 'Unlikely'} to purchase — estimated probability: {prob:.1%}")
