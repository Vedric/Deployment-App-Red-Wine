# Importation des libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


# Titre de l'application

st.write("""
# Application de prédiction de la qualité du vin rouge
Cette application prédit la ***Qualité du vin rouge*** !
""")

# Créer une barre latérale pour les fonctions de saisie de l'utilisateur

st.sidebar.header('Paramètres')

def parameter_adjustments():
        fixed_acidity = st.sidebar.slider('Acidité fixe', 4.6, 16.5, 4.6)
        volatile_acidity = st.sidebar.slider('Acidité volatile', 0.12,1.59 , 0.12)
        citric_acid = st.sidebar.slider('Acide citrique', 0.0,1.0 , 0.0)
        chlorides = st.sidebar.slider('Chlorures', 0.01,0.7 , 0.01)
        total_sulfur_dioxide=st.sidebar.slider('Dioxyde de soufre total', 6.0,291.0 , 6.0)
        alcohol=st.sidebar.slider('Alcool', 8.4,15.2, 8.4)
        sulphates=st.sidebar.slider('Sulfates', 0.33,2.0,0.33)
        data = {'Acidité fixe': fixed_acidity,
                'Acidité volatile': volatile_acidity,
                'Acide citrique': citric_acid,
                'Chlorures': chlorides,
              'Dioxyde de soufre':total_sulfur_dioxide,
              'Alcool':alcohol,
                'Sulfates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = parameter_adjustments()

st.subheader('Paramètres utilisateur')
st.write(df)

#Lecture du dataset

df_red_wine=pd.read_csv("winequality-red.csv")
X =np.array(df_red_wine[['fixed acidity', 'volatile acidity' , 'citric acid' ,
 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(df_red_wine['quality'])

#Random forest model

RandomF_Class = RandomForestClassifier()
RandomF_Class.fit(X, Y)
st.subheader('Notations')
st.write(pd.DataFrame({
   'wine quality': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]}))

prediction = RandomF_Class.predict(df)
prediction_proba = RandomF_Class.predict_proba(df)
st.subheader('Prédiction')
st.write(prediction)

st.subheader('Prediction (Probabilités)')
st.write(prediction_proba)
