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
dataset.index = np.arange(2, len(dataset) + 2)

model = joblib.load('simpleclassifier.joblib')


features_array = preprocessData(dataset)
st.title("andwemet")
st.write("These recommendations are generated based on your subjective answers, using AI that is private and secure. We think they'd be a great fit, but hey- no algorighm can ever know what's best for you")
st.write("All users:")
DatasetSlot = st.empty()
rowNumberSlot = st.empty()
findMyMatchSlot = st.empty()
# showMyProfileSlot = st.empty()
DisplayMatchesSlot = st.empty()

if "isError" not in st.session_state:
    st.session_state.isError = False

if(st.session_state.isError):
    DisplayMatchesSlot.write("The servers are crunching some new entries. Please try again later (PS: It's a good time to grab a cup of coffee)")

DatasetSlot.dataframe(dataset.astype(str))

if "matches" not in st.session_state:
    st.session_state.matches = None

if st.session_state.matches is not None:
    DisplayMatchesSlot.write(st.session_state.matches)

row = rowNumberSlot.number_input("Enter your row number and press enter", min_value=2, max_value=len(dataset)+1, step=1)
findMyMatchSlot.button(f"Find match for row: {row}", on_click=lambda: find_match(row-2))
# showMyProfileSlot.write(dataset.iloc[row].astype(str))


def find_match(row):
    features = features_array[row]
    features = features.reshape(1, -1)
    distances = []
    indices = []
    try:
        #  throw an error to test the error message
        throw_error = 1/0
        # distances, indices = model.kneighbors(features)
    except(Exception):
        st.session_state.isError = True
        return
    match_index = indices[0]
    data = dataset.iloc[match_index]
    data = data[data["My Gender"]!= dataset.iloc[row]["My Gender"]]

    if(dataset.iloc[row]["My Gender"]=="Female"):
        data = data[((data["My age"])>= (dataset.iloc[row]["My age"]-2)) & ((data["My age"])<= (dataset.iloc[row]["My age"]+5)) ]
        
    else:
        data = data[(data["My age"])<= (dataset.iloc[row]["My age"]+3)]
            
    st.session_state.matches = data
    DisplayMatchesSlot.write(st.session_state.matches)