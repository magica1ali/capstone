import streamlit as st
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

st.write(st.session_state.fig0) # Topic Information 
st.write(st.session_state.fig3) # Topic similarity