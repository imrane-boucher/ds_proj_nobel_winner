# ressources : https://github.com/dataprofessor/code/blob/master/streamlit/part3/penguins-app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# Economics Nobel Prize Prediction App
This app predicts whether an economist is likely to be rewarded with a nobel prize in the future!
Data obtained from the [IDEAD RePEc project](https://ideas.repec.org/top/top.person.alldetail.html).
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example Esther Duflo CSV input file](https://raw.githubusercontent.com/imrane-boucher/ds_proj_nobel_winner/master/examples/esther_duflo.csv)
""")
st.sidebar.markdown("""
[Example Darcon Acemoglu CSV input file](https://raw.githubusercontent.com/imrane-boucher/ds_proj_nobel_winner/master/examples/daron_acemoglu.csv)
""")
st.sidebar.markdown("""
[Example Matthew D. Shapiro CSV input file](https://raw.githubusercontent.com/imrane-boucher/ds_proj_nobel_winner/master/examples/matthew_d_shapiro.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", 
type=['csv'])


if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        nb_downl = st.sidebar.slider('Number of downloads', 0,10000,5000)
        nb_pages = st.sidebar.slider('Number of pages', 0,10000,5000)
        Students = st.sidebar.slider('Number of students', 0,100,10)
        nb_works = st.sidebar.slider('Number of works', 0,1500,200)
        h_index = st.sidebar.slider('h Score', 0,100,15)
        nb_cit = st.sidebar.slider('Number of citations', 1000,50000,10000)
        vn_award = st.sidebar.selectbox('Von Neumann Award Laureate', (1, 0))
        clark = st.sidebar.selectbox('Clark Medal Laureate', (1, 0))
        top10_shangai_yn = st.sidebar.selectbox('Top 10 Shanghai university y/n',(1,0))
        usa_yn = st.sidebar.selectbox('Work in  usa',(1, 0))
        descri_len = st.sidebar.slider('Description length', 0,30,10)
        len_work = st.sidebar.slider('Number of pages per work', 0,10,5)

        data = {'nb_downl': nb_downl,
                'nb_pages': nb_pages,
                'nb_stud': Students,
                'nb_works': nb_works,
                'h_index': h_index,
                'nb_cit': nb_cit,
                'vn_yn': vn_award,
                'clark_yn': clark,
                'top10_uni': top10_shangai_yn,
                'usa_yn': usa_yn,
                'descri_len': descri_len,
                'len_work': len_work}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

 
df_model = pd.read_csv('data/model_nobel.csv')
df_features = df_model.drop(columns=['nobel'])


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Reads in saved classification model
load_rfus = pickle.load(open('nobel_rfus.pkl', 'rb'))

# Apply model to make predictions
prediction = load_rfus.predict(input_df)
prediction_proba = load_rfus.predict_proba(input_df)


st.subheader('Prediction')
nobel_yn = np.array(['Non Nobel winner', 'Nobel winner'])
st.write(nobel_yn[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)