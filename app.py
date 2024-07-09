import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder 
from scipy.stats import mode
import numpy as np
from scipy import stats


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning, message="Unlike other reduction functions")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Support for non-numeric arrays")

# Load models
svm_model = joblib.load("./svc_model.joblib")
nb_model = joblib.load("./rf_model.joblib")
rf_model = joblib.load("./nb_model.joblib")

df=pd.read_csv("./datasets/Testing.csv").dropna(axis=1)
encoder=LabelEncoder()
encoder.fit_transform(df["prognosis"])

X=df.iloc[:,:-1]

symptoms =X.columns.values

symptom_index={}

for index,value in enumerate(symptoms):
    symptom= " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom]=index

data_dict={
    "symptoms_index":symptom_index,
    "predictions_classes":encoder.classes_
}

# Defining the Function 
# Input: string containing symptoms separated by commas 
# Output: Generated predictions by models

def predictdisease(symptoms):
    # Converting the input string to a list of symptoms and trimming spaces
    symptoms = [symptom.strip() for symptom in symptoms.split(",")]

    input_data = [0] * len(data_dict["symptoms_index"])
    for symptom in symptoms:
        index = data_dict["symptoms_index"].get(symptom)
        if index is not None:
            input_data[index] = 1
        else:
            # Handle case where symptom is not found in symptom_index
            raise KeyError(f"Symptom '{symptom}' not found in the dataset.")

    input_data = np.array(input_data).reshape(1, -1)
    
    # Predicting the disease using the trained model
    rf_predicton = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

    # Determine final prediction based on the most frequent prediction
    predictions = [rf_predicton, nb_prediction, svm_prediction]
    values, counts = np.unique(predictions, return_counts=True)
    final_prediction = values[np.argmax(counts)]

    return {
        "Random Forest": rf_predicton,
        "Naive Bayes": nb_prediction,
        "Support Vector Machine": svm_prediction,
        "final_prediction": final_prediction
    }



print(predictdisease("Vomiting"))


st.title("Disease Prediction Based on Symptoms 🧑‍⚕️")

symptoms_input = st.text_input("Enter symptoms separated by commas (e.g., 'Stomach Pain, Vomiting, Weight Loss')")

if st.button("Predict"):
    if symptoms_input:
        prediction = predictdisease(symptoms_input)
        st.write(f"Predicted Disease: {prediction['final_prediction']}")
        st.write(f"Random Forest: {prediction['Random Forest']}")
        st.write(f"Naive Bayes: {prediction['Naive Bayes']}")
        st.write(f"Support Vector Machine: {prediction['Support Vector Machine']}")
    else:
        st.warning("Please enter symptoms.")





