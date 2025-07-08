import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from IPython.display import HTML  # For animation handling

# Add this at the top for animated flower
st.markdown("""
<style>
.flower-animation {
    display: block;
    margin: 0 auto;
    max-width: 200px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()


flower_url = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExanY1MGd5enQ1aWkxMXR6cmw2ejEyaWE1NTk0dWN4bzVsOHk3NG56YyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/QiY6ZEhKYvOHnPgQXf/giphy.gif"  

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])


with st.sidebar:
    sl = st.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
    sw = st.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
    pl = st.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
    pw = st.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))


input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=df.columns[:-1])
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]


col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f'<img class="flower-animation" src="{flower_url}">', unsafe_allow_html=True)
    
with col2:
    st.subheader("Prediction")
    st.markdown(f"<h3 style='color: purple; font-size: xx-large;'>{predicted_species}</h3>", unsafe_allow_html=True)
