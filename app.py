# 1. Library imports
import pandas as pd
import logging
import streamlit as st
import urllib.request, json

DATA_URL = ('https://p7opencr.herokuapp.com/predict/')
local_data = pd.read_csv('clean_data1.zip',compression='zip')

def load_prediction(sk_id):
    with urllib.request.urlopen(DATA_URL + str(sk_id)) as url:
        data = json.loads(url.read().decode())
    real_data = float(data['message'].split(' ')[0].replace('[',''))
    return real_data

client_choice = st.sidebar.selectbox("Chose your client", local_data.SK_ID_CURR)

st.title('Bank stuff app')


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
data = load_prediction(client_choice)
data_load_state.text('Loading data...done!')

st.subheader('Client Informations')
st.metric("Percentage", data*100)

st.dataframe(local_data)



