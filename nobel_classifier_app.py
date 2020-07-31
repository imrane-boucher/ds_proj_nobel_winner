# ressources : https://github.com/dataprofessor/code/blob/master/streamlit/part3/penguins-app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Economics Nobel Prize Prediction App
This app predicts whether an economist is likely to be rewarded with a nobel prize in the future!
Data obtained from the [IDEAD RePEc project](https://ideas.repec.org/top/top.person.alldetail.html).
""")

st.sidebar.header('User Input Features')

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
print(input_df.head())
    # Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
df_model = pd.read_csv('data/model_nobel.csv')
df_features = df_model.drop(columns=['nobel'])
#df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#encode = ['top10_uni','usa_yn','clark_yn', 'vn_yn']
#for col in encode:
#    dummy = pd.get_dummies(input_df[col], prefix=col)
#    input_df = pd.concat([input_df,dummy], axis=1)
#    del input_df[col]
#df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

#if uploaded_file is not None:
#    st.write(df)
#else:
    #st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
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