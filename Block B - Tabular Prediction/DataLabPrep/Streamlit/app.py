import streamlit as st
st.title('My First Streamlit App')
st.write('Hello, Streamlit!')
number = st.slider('Pick a number', 0, 100)
st.write('You selected:', number)