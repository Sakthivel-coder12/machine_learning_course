import streamlit as st
import pandas as pd
import numpy as np

## https://streamlit.io/ refere all the command in this 
st.title("Hello Streamlit")

st.write("This is simple text")

df = {
    'name' : ['sakthivel','nithiya','faizan','nivasri'],
    'age'  : [19,19,19,9]
}
st.markdown(f"<h4>Here is the dataframe</h4>",unsafe_allow_html=True)
st.write(pd.DataFrame(df))


## creating line chart 

chart_data = pd.DataFrame(
    np.random.randn(20,3),columns=['A','B','C']
)
st.line_chart(chart_data)