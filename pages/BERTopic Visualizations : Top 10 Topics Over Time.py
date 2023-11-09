import streamlit as st

st.title("BERTopic Visualizations : Top 10 Topics Over Time")
st.header('Displays the top 10 topics by frequency over the timespan of inputted reports.')
st.write(st.session_state.fig5)