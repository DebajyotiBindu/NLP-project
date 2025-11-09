import joblib
import streamlit as st
model=joblib.load('role_predictor_model.pkl')
tv=joblib.load('tfidf_vectorizer.pkl')
label_enc=joblib.load('label_encoder.pkl')

st.title("Role Predictor")

text=st.text_area("Enter the job description here")

if st.button("Predict role"):
    if text.strip():
        vect_text=tv.transform([text])
        pred_role=model.predict(vect_text)
        result=label_enc.inverse_transform(pred_role)[0]

        st.success(f"The predicted role is: {result}")
    else:
        st.error("Please enter a valid job description.")
