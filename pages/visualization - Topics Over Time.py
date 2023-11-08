import streamlit as st
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

st.title("BerTopic Visualizations : Topics Over Time")

st.write(st.session_state.fig5) # Topics Over Time