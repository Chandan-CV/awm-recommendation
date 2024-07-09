import joblib
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import get_features_array, preprocessData
dataurl = 'https://docs.google.com/spreadsheets/d/1EmeVCJzCiMISggPuS5clv-gz4mznq24NMyFV8aXkbK4/edit?gid=0#gid=0'
dataurl = dataurl.replace('/edit?gid=', '/export?format=csv&gid=')
dataset = pd.read_csv(dataurl)
# dataset = dataset[dataset['My age'] != 'test']
dataset = dataset.fillna('')

model = joblib.load('simpleclassifier.joblib')


features_array = preprocessData(dataset)
st.title("andwemet")
st.write("This recommendations are generated based on your subjective answers, using AI that is private and secure. We think they'd be a great fit, but hey- no algorighm can ever know what's best for you")
st.write("All users:")
DatasetSlot = st.empty()
rowNumberSlot = st.empty()
findMyMatchSlot = st.empty()
# showMyProfileSlot = st.empty()
DisplayMatchesSlot = st.empty()


DatasetSlot.dataframe(dataset.astype(str))

if "matches" not in st.session_state:
    st.session_state.matches = None

if st.session_state.matches is not None:
    DisplayMatchesSlot.write(st.session_state.matches)

row = rowNumberSlot.number_input("Enter your row number and press enter", min_value=0, max_value=len(dataset)-1, step=1)
findMyMatchSlot.button(f"Find match for row: {row}", on_click=lambda: find_match(row))
# showMyProfileSlot.write(dataset.iloc[row].astype(str))


def find_match(row):
    features = features_array[row]
    features = features.reshape(1, -1)
    distances, indices = model.kneighbors(features)
    match_index = indices[0]
    data = dataset.iloc[match_index]
    data = data[data["My Gender"]!= dataset.iloc[row]["My Gender"]]
    st.session_state.matches = data
    DisplayMatchesSlot.write(st.session_state.matches)