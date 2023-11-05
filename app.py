import streamlit as st
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
                # Load a pre-trained Sentence Transformer model
                model = SentenceTransformer("all-mpnet-base-v2")
                # Function to instantiate BERTopic Model
                topic_model = BERTopic.load("magica1/saxa3-capstone",model)
                st.write('Model instantiated!')
                
                time.sleep(1)
                
                #st.write('Fitting data to model and extracting topics...')
                #topics, probs = topic_model.transform(filtered_sentences)
                example = "Stalemate is a drawn position. It doesn't matter who has captured more pieces or is in a winning position"
                topic, prob = topic_model.transform(example)
                #st.write('Model fitted!')
                
                time.sleep(1)
                
                # custom_funcs.plot_topics_over_time(topic_model, translated_text, timestamps)
                # custom_funcs.show_doc_info(topic_model, translated_text)

                st.write('Creating Visualizations...')
                status.update(label="Process complete!", state="complete", expanded=False)
            
                st.session_state.fig0 = topic_model.get_topic_info(topic)
                st.session_state.fig1 = topic_model.visualize_hierarchy()
                st.session_state.fig2 = topic_model.visualize_topics()
                st.session_state.fig3 = topic_model.visualize_heatmap()
                st.session_state.fig4 = topic_model.visualize_barchart()
            custom_funcs.prove_success_func(topic_model)

        except Exception as e:
            # Catch any exceptions and display them using st.exception()
            st.exception(e)


if __name__ == '__main__':
    main()
