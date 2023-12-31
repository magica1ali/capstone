import streamlit as st
import pandas as pd 
import numpy as np 
import time
import custom_funcs
from bertopic import BERTopic
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
                st.write(len(model_timestamp))
                
                # Load the text file(sentences used in model) into a DataFrame 
                sentences_list = []
                sentence_path = "data/sentences.txt"
                # Open the file and read the sentences
                with open(sentence_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # Remove leading/trailing whitespaces and append to the list
                        sentences_list.append(line.strip())
                # Combine the two lists
                combined_list = sentences_list + filtered_sentences
                num_reccomendations_processed = st.write(len(sentences_list))
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
            topics_over_time = topic_model.topics_over_time(docs=sentences_list,
                                                timestamps=model_timestamp)
            st.write(topic_model.visualize_topics_over_time(topics_over_time))
            # custom_funcs.plot_topics_over_time(topic_model, translated_text, timestamps)
            # custom_funcs.show_doc_info(topic_model, translated_text)
            custom_funcs.prove_success_func(topic_model)

        except Exception as e:
            # Catch any exceptions and display them using st.exception()
            st.exception(e)


if __name__ == '__main__':
    main()
