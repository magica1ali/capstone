import streamlit as st
import time
import custom_funcs
import nltk
import spacy
from bertopic import BERTopic
import matplotlib.pyplot as plt

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
                st.write('PDFs ingested!')

                time.sleep(1)

                st.write('Parsing, spell-checking, and preprocessing recommendations text from PDFs...')
                # Function to parse out the recommendations section, clean, and preprocess corpus
                corpus,translated_text = custom_funcs.parse_clean_func(text_dict)
                st.write(len(translated_text))

                text_year = custom_funcs.spacyLayer(translated_text,corpus)
                filtered_sentences = custom_funcs.append_years(text_year)
                timestamps = custom_funcs.datetime_layer(filtered_sentences)
                st.write('Reccomendations text preprocessed!')
                st.write(len(filtered_sentences))
                st.write(filtered_sentences[0])
                
                time.sleep(1)
                
                st.write('Instantiating BERTopic Model...')
                # Function to instantiate BERTopic Model
                topic_model,topics, probs = custom_funcs.bertopic_model_text(filtered_sentences)
                st.write('Model instantiated!')
                
                time.sleep(1)
                
                #st.write('Fitting data to model and extracting topics...')
                #topics, probs = custom_funcs.topic_model.fit_transform(filtered_sentences)
                #st.write('Model fitted!')
                
                time.sleep(1)
                
                st.write('Creating Visualizations...')
                fig1 = custom_funcs.topic_model.visualize_hierarchy()
                st.write(fig1)
                # custom_funcs.plot_topics_over_time(topic_model, translated_text, timestamps)
                # custom_funcs.show_doc_info(topic_model, translated_text)
                
                status.update(label="Process complete!", state="complete", expanded=False)
            
            custom_funcs.prove_success_func(topic_model)

        except Exception as e:
            # Catch any exceptions and display them using st.exception()
            st.exception(e)


if __name__ == '__main__':
    main()
