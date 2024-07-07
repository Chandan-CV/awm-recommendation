import joblib
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import get_features_array
dataurl = 'https://docs.google.com/spreadsheets/d/1EmeVCJzCiMISggPuS5clv-gz4mznq24NMyFV8aXkbK4/edit?gid=0#gid=0'
dataurl = dataurl.replace('/edit?gid=', '/export?format=csv&gid=')
dataset = pd.read_csv(dataurl)
# dataset = dataset[dataset['My age'] != 'test']
dataset = dataset.fillna('')

model = joblib.load('v1_knn.joblib')


features_array = get_features_array(dataset)
st.title("AndWeMet")

DatasetSlot = st.empty()
rowNumberSlot = st.empty()
findMyMatchSlot = st.empty()
showMyProfileSlot = st.empty()
DisplayMatchesSlot = st.empty()


DatasetSlot.dataframe(dataset.astype(str))

if "matches" not in st.session_state:
    st.session_state.matches = None

if st.session_state.matches is not None:
    DisplayMatchesSlot.write(st.session_state.matches)

# def find_match():
#     global row
#     global slot4
#     features = features_array.getrow(row)
#     features = features.reshape(1, -1)
#     distances, indices = model.kneighbors(features)
#     match_index = indices[0]
#     data = dataset.iloc[match_index]
#     data = data[data["My Gender"] != dataset.iloc[row]["My Gender"]]
#     slot4.write(data)


# row = slot2.text_input("Enter the row number")
# if(row != ''):
#     row = int(row)
#     rowToDisplay = dataset.iloc[row].astype(str)
#     slot3.button("Find the match", on_click=find_match)
#     slot4.dataframe(rowToDisplay)

row = rowNumberSlot.number_input("Enter the row number", min_value=0, max_value=len(dataset)-1, step=1)
findMyMatchSlot.button("Find my match", on_click=lambda: find_match(row))
showMyProfileSlot.write(dataset.iloc[row].astype(str))
def find_match(row):
    features = features_array.getrow(row)
    features = features.reshape(1, -1)
    distances, indices = model.kneighbors(features)
    match_index = indices[0]
    data = dataset.iloc[match_index]
    data = data[data["My Gender"]!= dataset.iloc[row]["My Gender"]]
    st.session_state.matches = data
    DisplayMatchesSlot.write(st.session_state.matches)