import streamlit as st
import pandas as pd 
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def main():

    st.title("PDF Text Extraction")

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
                num_pdfs_ingested = len(text_dict)
                st.write(f'{num_pdfs_ingested} PDF(s) ingested!')

                time.sleep(1)

                st.write('Parsing, spell-checking, and preprocessing recommendations text from PDFs...')
                # Function to parse out the recommendations section, clean, and preprocess corpus
                corpus,translated_text = custom_funcs.parse_clean_func(text_dict)
                text_year = custom_funcs.spacyLayer(translated_text,corpus)
                filtered_sentences = custom_funcs.append_years(text_year)
                
                # Load a CSV file as a DataFrame of pretrained model's timestamps
                df = pd.read_csv('data/datetime_index.csv')
                # Convert the "timestamp" to a DatetimeIndex
                df['timestamps'] = pd.to_datetime(df['timestamps']) 
                model_timestamp = df.set_index('timestamps').index
                timestamps = custom_funcs.datetime_layer(filtered_sentences)
                # Concatenate the two DatetimeIndex objects
                combined_timestamps = model_timestamp.union(timestamps)
                
                # Create a file uploader widget
                uploaded_file = st.file_uploader("Upload a file containing sentences", type=["txt"])

                # Initialize an empty list to store the sentences
                sentences = []

                if uploaded_file is not None:
                # Read and process the uploaded file
                    contents = uploaded_file.read()
                # Split the contents into a list of sentences (assuming each sentence is on a new line)
                    sentences = contents.split("\n")
                sentences.extend(filtered_sentences)
                num_reccomendations_processed = st.write(len(sentences))
                st.write(f'{num_reccomendations_processed} Document(s) cleaned and preprocessed!')
                
                
                time.sleep(1)
                
                st.write('Instantiating BERTopic Model...')
                # Load a pre-trained Sentence Transformer model
                model = SentenceTransformer("all-mpnet-base-v2")
                # Function to instantiate BERTopic Model
                topic_model = BERTopic.load("magica1/saxa3-capstone",model)
                st.write('Model instantiated!')
                
                time.sleep(1)
                
                #st.write('Fitting data to model and extracting topics...')
                topics, probs = topic_model.transform(filtered_sentences)
                #st.write('Model fitted!')
                
                time.sleep(1)

                st.write('Creating Visualizations...')
                
                status.update(label="Process complete!", state="complete", expanded=False)

            #generate_visualizations_func(topic_model, timestamps, filtered_sentences)
            topics_over_time = topic_model.topics_over_time(docs=sentences,
                                                timestamps=combined_timestamps,
                                                global_tuning=True,
                                                evolution_tuning=True,
                                                nr_bins=15)
            st.write(topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10))
            # custom_funcs.plot_topics_over_time(topic_model, translated_text, timestamps)
            # custom_funcs.show_doc_info(topic_model, translated_text)
            custom_funcs.prove_success_func(topic_model)

        except Exception as e:
            # Catch any exceptions and display them using st.exception()
            st.exception(e)


if __name__ == '__main__':
    main()
