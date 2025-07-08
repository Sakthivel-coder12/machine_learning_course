import streamlit as st
import numpy as np
import pandas as pd


st.title("The text input form")

st.write("Enter the text")
st.text_area("Enter the text")

name = st.text_input("Enter the name",placeholder="give you name")
if name:
    st.write("Welcome to the course :" , name)

age = st.slider("Select you age",0,100,25)

if age:
    st.write("The age you are selected is : ",age)

options = ['python','java','c++','c','html','css','javascript']
choice = st.selectbox("Choose the options :",options)
st.write("The select option is :",choice)

df = pd.DataFrame({
    'name' : ['sakthivel','nithiya','faizan','nivasri'],
    'age'  : [19,19,19,9]
})
st.subheader("The dataframe is : ")

st.write(df)
df.to_csv("sample.csv")
upload_file = st.file_uploader("Choose the file",type="csv")

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write(df)