# 1. Library imports
import pandas as pd
import logging
import streamlit as st
import urllib.request, json
import joblib
import shap
import streamlit.components.v1 as components
import matplotlib

st.set_option('deprecation.showPyplotGlobalUse', False)
DATA_URL = ('https://p7opencr.herokuapp.com/predict/')
local_data = pd.read_csv('clean_data1.zip',compression='zip')
pipeline = joblib.load('model.joblib')
model = pipeline.named_steps['mdl']
X = local_data.drop('TARGET', axis=1)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
global_shap_values = explainer(X)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def load_prediction(sk_id):
    with urllib.request.urlopen(DATA_URL + str(sk_id)) as url:
        data = json.loads(url.read().decode())
    real_data = float(data['message'].split(' ')[0].replace('[',''))
    return real_data

client_choice = st.sidebar.selectbox("Chose your client", local_data.SK_ID_CURR)

st.title('Prêt à dépenser - Scoring Crédit')


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
data = load_prediction(client_choice)
data_load_state.text('Loading data...done!')

st.subheader('Client Informations')
st.metric("Predicted Score", str(data*100))

st.dataframe(local_data)

# visualize the training set predictions
#st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
index = local_data.index[local_data['SK_ID_CURR'] == client_choice].tolist()[0]
st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][index,:], features = X.iloc[index,:]))
global_shap_values.values=global_shap_values.values[:,:,1]
global_shap_values.base_values=global_shap_values.base_values[:,1]

#st_shap(shap.plots.waterfall((global_shap_values.base_values[0], global_shap_values[0][0], X[0])))
a = explainer.expected_value[0]
b = shap_values[0][index]
c = X.iloc[index,:]
#st_shap(shap.plots._waterfall.waterfall_legacy(a, b, c))
# visualize the training set predictions
st.pyplot(fig=shap.plots._waterfall.waterfall_legacy(a, b, c))

#st.pyplot(fig=shap.summary_plot(shap_values[1], features=X, max_display=10))



