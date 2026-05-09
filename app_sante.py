import streamlit as st

# ---------------------------------------------------
# CONFIGURATION (DOIT ÊTRE EN PREMIER)
# ---------------------------------------------------
st.set_page_config(
    page_title="Prédicteur du nombre de médecins",
    page_icon="👨🏾‍⚕️",
    layout="centered"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ---------------------------------------------------
# CHARGEMENT DATASET
# ---------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("sante_cameroun.csv")
    data = data.drop_duplicates()
    return data

# ---------------------------------------------------
# CHARGEMENT MODELE
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open("model_sante.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------------------------------
# APPLICATION
# ---------------------------------------------------
def main():

    st.markdown(
        "<h1 style='text-align:center;color:brown;'>⚕️Santé Cameroun App</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align:center;color:black;'>Prédiction du nombre de médecins</h2>",
        unsafe_allow_html=True
    )

    menu = ["Home", "Analysis", "Data Visualisation", "Machine Learning"]
    choice = st.sidebar.selectbox("Select Menu", menu)

    data = load_data()

    # ---------------------------------------------------
    # HOME
    # ---------------------------------------------------
    if choice == "Home":

        st.subheader("Présentation du projet")

        st.write("""
        Cette application prédit le nombre de médecins dans un district
        sanitaire au Cameroun à partir d'indicateurs socio-sanitaires.

        Le modèle utilisé est un XGBoost Regressor entraîné sur le dataset
        sante_cameroun.csv.
        """)

        st.subheader("Variables du dataset")

        st.write(pd.DataFrame({
            "Variable": [
                "revenu_moyen",
                "acces_sante",
                "taux_vaccination",
                "pollution",
                "taux_mortalite_infantile",
                "nombre_medecins"
            ],
            "Type": [
                "Numérique",
                "Numérique",
                "Numérique",
                "Numérique",
                "Numérique",
                "Numérique"
            ],
            "Description": [
                "Revenu moyen du district (FCFA)",
                "Taux d'accès aux soins (%)",
                "Taux de vaccination (%)",
                "Indice de pollution",
                "Taux de mortalité infantile",
                "Nombre de médecins (variable cible)"
            ]
        }))

    # ---------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------
    elif choice == "Analysis":

        st.subheader("Dataset Santé Cameroun")
        st.write(data.head())

        if st.checkbox("Résumé statistique"):
            st.write(data.describe())

        if st.checkbox("Valeurs manquantes"):
            st.write(data.isnull().sum())

        if st.checkbox("Corrélation"):
            fig, ax = plt.subplots(figsize=(10, 7))

            sns.heatmap(
                data.select_dtypes(include="number").corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax
            )

            st.pyplot(fig)
            plt.close(fig)

    # ---------------------------------------------------
    # DATA VISUALISATION
    # ---------------------------------------------------
    elif choice == "Data Visualisation":

        if st.checkbox("Distribution — nombre de médecins"):

            fig, ax = plt.subplots(figsize=(8, 4))

            sns.histplot(
                data["nombre_medecins"],
                bins=20,
                ax=ax
            )

            ax.set_title("Distribution du nombre de médecins")

            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Revenu moyen vs Médecins"):

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.scatterplot(
                x="revenu_moyen",
                y="nombre_medecins",
                data=data,
                alpha=0.6,
                ax=ax
            )

            ax.set_title("Revenu moyen vs Nombre de médecins")

            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Accès aux soins vs Médecins"):

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.scatterplot(
                x="acces_sante",
                y="nombre_medecins",
                data=data,
                alpha=0.6,
                ax=ax
            )

            ax.set_title("Accès aux soins vs Nombre de médecins")

            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Vaccination vs Médecins"):

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.scatterplot(
                x="taux_vaccination",
                y="nombre_medecins",
                data=data,
                alpha=0.6,
                ax=ax
            )

            ax.set_title("Taux de vaccination vs Nombre de médecins")

            st.pyplot(fig)
            plt.close(fig)

    # ---------------------------------------------------
    # MACHINE LEARNING
    # ---------------------------------------------------
    elif choice == "Machine Learning":

        st.subheader("Prédiction du nombre de médecins")

        model = load_model()

        with st.form("prediction_form"):

            revenu_moyen = st.slider(
                "Revenu moyen (FCFA)",
                100000,
                1200000,
                500000,
                step=5000
            )

            acces_sante = st.slider(
                "Accès aux soins (%)",
                0.0,
                100.0,
                60.0,
                step=0.5
            )

            taux_vaccination = st.slider(
                "Taux de vaccination (%)",
                0.0,
                100.0,
                70.0,
                step=0.5
            )

            pollution = st.slider(
                "Indice de pollution",
                0.0,
                100.0,
                30.0,
                step=0.1
            )

            taux_mortalite_infantile = st.slider(
                "Mortalité infantile",
                -10.0,
                40.0,
                10.0,
                step=0.1
            )

            submit = st.form_submit_button(
                "Prédire le nombre de médecins"
            )

        if submit:

            input_data = pd.DataFrame([{
                "revenu_moyen": revenu_moyen,
                "acces_sante": acces_sante,
                "taux_vaccination": taux_vaccination,
                "pollution": pollution,
                "taux_mortalite_infantile": taux_mortalite_infantile
            }])

            st.write("Caractéristiques saisies :")
            st.write(input_data)

            prediction = model.predict(input_data)[0]

            st.success(
                f"Nombre de médecins estimé : {int(round(prediction))}"
            )


# ---------------------------------------------------
# LANCEMENT
# ---------------------------------------------------
if __name__ == "__main__":
    main()
