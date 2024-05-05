import streamlit as st
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf


def transf(fli_pred):
    # Variables numéricas y categóricas
    numeric_features = fli_pred.select_dtypes(
        include=['float64', 'int64']).columns
    categorical_features = fli_pred.select_dtypes(include=['object']).columns

    # Numéricas
    numeric_preprocessor = StandardScaler()
    X_numeric = numeric_preprocessor.fit_transform(fli_pred[numeric_features])

    # Categóricas
    categorical_preprocessor = OneHotEncoder()
    X_categorical = categorical_preprocessor.fit_transform(
        fli_pred[categorical_features])

    # Concatenar
    X_preprocessed = np.hstack((X_numeric, X_categorical.toarray()))
    return X_preprocessed


def predict_formato(y_pred_2024):
    # Formato predicción
    y_pred_2024 = np.round(y_pred_2024, fli_pred)
    fli_pred['Passengers'] = pd.Series([x[0] for x in y_pred_2024])
    # Sobreventa de predicción
    fli_pred.loc[fli_pred['Passengers'] > fli_pred['Capacity'],
                 'Passengers'] = fli_pred['Capacity']
    return fli_pred


df_up = st.file_uploader("Archivo csv", type=["csv"])

loaded_model1 = keras.saving.load_model('Modelos/Modelo_Cap1')
loaded_model2 = keras.saving.load_model('Modelos/Modelo_Cap2')


if (df_up is not None):
    # %%% Leer archivos cargados
    fli_pred = pd.read_csv(df_up)

    # Predicción
    X_preprocessed = transf(fli_pred)
    y_pred_2024_1 = loaded_model1.predict(X_preprocessed)
    y_pred_2024_2 = loaded_model2.predict(X_preprocessed)

    fli_pred_1 = predict_formato(y_pred_2024_1, fli_pred)
    fli_pred_2 = predict_formato(y_pred_2024_2, fli_pred)

    fli_pred_1 = fli_pred_1[fli_pred_1['Capacity'] <= 186]
    fli_pred_2 = fli_pred_2[fli_pred_2['Capacity'] > 186]

    fli_pred_tot = pd.concat([fli_pred_1, fli_pred_2])
    st.write(fli_pred_tot)

    st.download_button(
        label="Predicciones",
        data=pd.to_csv(fli_pred_tot, index=False),
        file_name=f"{df_up.name.replace('.csv','')}_predict.xlsx")
