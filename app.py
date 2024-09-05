import pandas as pd
import streamlit as st 
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Insurance Charge Predictions")

def get_model():
    return load_model('insurance_model')

def predict(model, data):
    prediction = predict_model(model, data=data)
    print(prediction.head())  # Print to inspect the structure
    
    # Adjust this line based on the actual structure of your prediction dataframe
    label_col = prediction.columns[-1]  # Assuming the label column is the last column
    
    return prediction.iloc[0][label_col]


model = get_model()

st.title("Insurance Charges Prediction")

form = st.form('charges')
age = form.number_input('Age', min_value=1, max_value=100, value=25)
sex = form.radio("Sex", ['Male', 'Female'])
bmi = form.number_input('BMI', min_value=1.0, max_value=50.0, value=20.0)  # Ensure all values are floats
children = form.slider('Children', min_value=1, max_value=10, value=0)
region_list = ['Southwest', 'Northwest', 'Southeast', 'Northeast']  # Capitalize regions for consistency
region = form.selectbox('Region', region_list)

if form.checkbox('Smoker'):
    smoker = 'yes'
else:
    smoker = 'no'
    
predict_button = form.form_submit_button('Predict')

input_dict = {
    'age': age,
    'sex': sex.lower(),
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region.lower()
}

input_df = pd.DataFrame([input_dict])

if predict_button:
    output = predict(model, input_df)
    st.success('The predicted charges are ${:.2f}'.format(output))