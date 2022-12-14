# 1. Library imports
import pandas as pd
import streamlit as st
import urllib.request, json
#import joblib
import shap
import streamlit.components.v1 as components
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
DATA_URL = ('https://p7opencr.herokuapp.com/predict/')
local_data = pd.read_csv('clean_data1.zip',compression='zip')
X = local_data.drop('TARGET', axis=1)
threshold = 0.062

with open('shap_values_0.npy', 'rb') as f:
    shap_values_0 = np.load(f)

explainer_expected_value = [2.641399018192998, -2.641399018192998]

@st.cache
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


@st.cache
def load_prediction(sk_id):
    with urllib.request.urlopen(DATA_URL + str(sk_id)) as url:
        data = json.loads(url.read().decode())
    real_data = (float(data['message'].split(' ')[0].replace('[','')),float(data['message'].split(' ')[1].replace(']','')))
    return real_data

def createSubheader(pred_1):
    if(pred_1>threshold):
        st.markdown('<p style="color:Red;font-size:25px;font-weight: bold;">Pr??t Refus??</p>',  unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:Green;;font-size:25px;font-weight: bold;">Pr??t Accept??</p>',  unsafe_allow_html=True)

client_choice = st.sidebar.selectbox("Choisissez votre client", local_data.SK_ID_CURR)
st.sidebar.subheader("D??tail par variable")
variable_choice = st.sidebar.selectbox("Choisissez votre variable", local_data.columns)


st.title('Pr??t ?? d??penser - Scoring Cr??dit')

data = load_prediction(client_choice)

createSubheader(data[1])
st.metric("Score Pr??dit", '%.2f /100' % (data[0]*100))

st.dataframe(local_data)

st.subheader("Explication Locale des Features")
# visualize the training set predictions
#st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
index = local_data.index[local_data['SK_ID_CURR'] == client_choice].tolist()[0]
st_shap(shap.force_plot(explainer_expected_value[0], shap_values_0[index,:], features = X.iloc[index,:]))
#global_shap_values.values=global_shap_values.values[:,:,1]
#global_shap_values.base_values=global_shap_values.base_values[:,1]

#st_shap(shap.plots.waterfall((global_shap_values.base_values[0], global_shap_values[0][0], X[0])))
a = explainer_expected_value[0]
b = shap_values_0[index]
c = X.iloc[index,:]
#st_shap(shap.plots._waterfall.waterfall_legacy(a, b, c))
# visualize the training set predictions
st.pyplot(fig=shap.plots._waterfall.waterfall_legacy(a, b, c))
st.subheader("Explication Globale des Features")
st.pyplot(shap.summary_plot(shap_values_0, X))

#st.pyplot(fig=shap.summary_plot(shap_values[1], features=X, max_display=10))

fig, ax = plt.subplots()
#plt.xlim(min(local_data[variable_choice]),max(local_data[variable_choice]))
hist_data = ""
bins = 10
nb_unique = len(local_data[variable_choice].unique())
if nb_unique < 10:
    bins = nb_unique
    hist_data = local_data[variable_choice]
else:
    hist_data = local_data.loc[~is_outlier(local_data[variable_choice]),variable_choice]




#The sidebar histogram
ax.hist(hist_data,color = 'lightblue',edgecolor = 'black')
xvalue = local_data.loc[local_data['SK_ID_CURR'] == client_choice, variable_choice].values
ax.axvline(x=xvalue, color='red')

st.sidebar.pyplot(fig)
#st.text(shap_values)

#with open("shap_values_0.npy", 'wb') as f:
#np.save(f, shap_values_0)



del local_data, X
del a, b, c
