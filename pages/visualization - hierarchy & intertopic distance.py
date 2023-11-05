import streamlit as st
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

st.title("BerTopic Visualizations : Hierarchy and Interopic Distance")

st.write(st.session_state.fig1) # Hierarchy Chart
st.write(st.session_state.fig2) # Intertopic Distance 