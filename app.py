import streamlit as st
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
from io import BytesIO
import requests

def main():


    #Import lovely Georgetown University logo
    @st.cache_data
    def get_image():
        url = "https://msb.georgetown.edu/wp-content/uploads/2022/08/GU_MSB_Transparent_Vertical_Logo.png"
        r = requests.get(url)
        return BytesIO(r.content)
    
    st.image(get_image(),)
    
    st.title("PDF Topic Modeling Tool")
    st.header("Designed for the Advisory Committee on Women Veterans - U.S. Dept. of Veterans Affairs")
    st.subheader("By Saxa Capstone Team 3: \nMatthew Booth, Aarika Cox, Mike Halsema, Ali Mohamed, Logan Suba, and Su Tellakat")

    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:    
        
        # Wrap the entire code in a try-except block
        try:
            with st.status("Modeling topics from PDFs...",expanded=True) as status:
                #function to download dependencies
                st.write("Setting up environment...")
                custom_funcs.setup_func()
                st.write('Environment set up!')

                time.sleep(1)

                st.write("Ingesting PDFs...")
                text_dict = custom_funcs.process_pdfs(uploaded_files)
                st.write('PDFs ingested!')

                time.sleep(1)

                st.write('Parsing, spell-checking, and preprocessing recommendations text from PDFs...')
                # Function to parse out the recommendations section, clean, and preprocess corpus
                translated_text = custom_funcs.parse_clean_func(text_dict)

                timestamps = custom_funcs.datetime_layer(translated_text)
                st.write('Recommendations text preprocessed!')
                
                time.sleep(1)
                
                st.write('Instantiating BERTopic Model...')
                # Load a pre-trained Sentence Transformer model
                model = SentenceTransformer("all-mpnet-base-v2")
                # Function to instantiate BERTopic Model
                topic_model = BERTopic.load("magica1/saxa3-capstone",model)
                st.write('Model instantiated!')
                
                time.sleep(1)
                
                st.write('Fitting data to model and extracting topics...')
                topics, probs = topic_model.transform(translated_text)
                st.write('Model fitted!')
                
                time.sleep(1)

                st.write('Creating Visualizations...')
                status.update(label="Process complete!", state="complete", expanded=False)
            
                st.session_state.fig0 = topic_model.get_topic_info()
                st.session_state.fig1 = topic_model.visualize_hierarchy(top_n_topics = 10)
                st.session_state.fig2 = topic_model.visualize_topics(top_n_topics = 10)
                st.session_state.fig3 = topic_model.visualize_heatmap(top_n_topics = 10)
                st.session_state.fig4 = topic_model.visualize_barchart(top_n_topics = 10)
                st.session_state.fig5 = custom_funcs.generate_topics_over_time_func(topic_model, timestamps, topics)
            
            custom_funcs.prove_success_func(topic_model)


        except Exception as e:
            # Catch any exceptions and display them using st.exception()
            st.exception(e)


if __name__ == '__main__':
    main()
