import streamlit as st
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

st.title("BerTopic Visualizations : Top 10 Topics Over Time")
st.header('Displays the top 10 topics by frequency over the timespan of inputted reports')


# Create a slider widget to control the number of topics to include
num_topics = st.slider("Number of Topics to Include", 1, 10)
    
st.dataframe(st.session_state.topic_info)
st.write(st.session_state.topic_info['Name'])
st.write(st.session_state.fig5)