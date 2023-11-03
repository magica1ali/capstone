import streamlit as st
import datetime
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
import fitz
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
import pkg_resources
from symspellpy import SymSpell
import matplotlib.pyplot as plt

def setup_func():
    with st.spinner("Downloading resources..."):
        try:
            # Download NLTK resources
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

            st.empty()  # Clear the spinner
            
        
        except Exception as e:
            st.error(f"Error downloading resources: {e}")
            st.empty()  # Clear the spinner

#define pdf reading function
def process_pdfs(uploaded_files):
    text_dict = {}

    for pdf_file in uploaded_files:
        pdf_data = pdf_file.read()  # Read the PDF data
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        concatenated_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if re.search(r'table of contents', page_text, re.IGNORECASE):
                continue
            concatenated_text += page_text

        # Extract the report name using regex pattern
        pattern = r'\d{4}\sACWV\sReport'
        matched_name = re.search(pattern, pdf_file.name)
        report_name = matched_name.group()

        # Store the concatenated text in the dictionary with the report name as the key
        text_dict[report_name] = concatenated_text
    return text_dict

#define function to extract reocmmendations section from each PDF
def parse_clean_func(text_dict):
    # Sample list of document texts
    extracted_sections = {}
    original_texts = {}

    def remove_outreach(text):
        # Find the index of "Outreach" in the text
        outreach_index = text.lower().find("outreach")

        if outreach_index != -1:
            # Extract content after "Outreach"
            cleaned_text = text[outreach_index:]
            return cleaned_text
        else:
            return text

    def remove_rationale(text):
        # Define the regular expression pattern
        pattern = re.compile(r"rationale:(.*?)(comment:|$)", re.DOTALL)

        # Remove rationale sections
        text_without_rationale = re.sub(pattern, "comment:", text)

        return text_without_rationale

    def remove_members(text):
        # Find the index of "Members" in the text
        members_index = text.lower().find("members")

        if members_index != -1:
            # Extract content after "Outreach"
            cleaned_text = text[members_index:]
            return cleaned_text
        else:
            return text

    documents = text_dict.items()

    # Define a regular expression pattern to match section headings
    pattern = r'Part\s*[IVX]+\s+.*'  # Pattern to match headings like "PART I" or "PART II: Heading"

    # Loop through the list of document texts
    for document_name, document_text in documents:
        # Exception handling for reports from the 90s with a different format
        if re.search(r'199\d', document_name, re.IGNORECASE):
            pattern_part_iii = r'part iii \n'
            # Search for the pattern in the text
            match = re.search(pattern_part_iii, document_text.lower())
            start_index = match.start()
            end_index = document_text.lower().find('part iv', start_index)  # Case-insensitive search
            #1996 report
            if start_index != -1 and end_index != -1:
                #remove outreach, rationale, and comments sections using previously defined function
                recommendations_section = document_text[start_index:end_index].strip()
                recommendations_section = recommendations_section.lower()
                recommendations_section = remove_rationale(remove_outreach(recommendations_section))
                comment_pattern = r'comment:.*? [a-z]\.'
                cleaned_text = re.sub(comment_pattern, '', recommendations_section, flags=re.DOTALL)
                extracted_sections[document_name] = recommendations_section
                original_texts[document_name] = document_text
            #1998 report (didnt have a part iv)
            if start_index != -1 and end_index == -1:
            #remove outreach, rationale, and comments sections using previously defined function
                recommendations_section = document_text[start_index:len(document_text)].strip()
                recommendations_section = recommendations_section.lower()
                recommendations_section = remove_rationale(remove_outreach(recommendations_section))
                comment_pattern = r'comment:.*? [a-z]\.'
                cleaned_text = re.sub(comment_pattern, '', recommendations_section, flags=re.DOTALL)
                extracted_sections[document_name] = recommendations_section
                original_texts[document_name] = document_text
        elif re.search(r'1998', document_name, re.IGNORECASE):
            start_index = document_text.lower().find('part ii')  # Case-insensitive search
            end_index = document_text.lower().find('part iii', start_index)  # Case-insensitive search
        if start_index != -1 and end_index != -1:
            recommendations_section = document_text[start_index:end_index].strip()
            #remove members and outreach sections
            recommendations_section = remove_members(recommendations_section)
            recommendations_section = recommendations_section.lower()
            extracted_sections[document_name] = recommendations_section
            original_texts[document_name] = document_text
        # Exception handling for reports from 2002 (wild year for this report)
        elif '2002' in document_name:
            # Process the 2002 report differently (e.g., extract sections between "Recommendations and Rationale" and "VA Response to Recommendations")
            start_index = document_text.lower().find('recommendations and rationale')  # Case-insensitive search
            end_index = document_text.lower().find('va response to recommendations', start_index)  # Case-insensitive search
            if start_index != -1 and end_index != -1:
                recommendations_section = document_text[start_index:end_index].strip()
                recommendations_section = recommendations_section.lower()
                extracted_sections[document_name] = recommendations_section
                original_texts[document_name] = document_text

        else:
            # Find all matching headings in the document text for other files
            headings = re.findall(pattern, document_text, re.IGNORECASE)

            # Find the start and end index of the 'RECOMMENDATIONS' section for other files
            start_index = None
            end_index = None

            for i, heading in enumerate(headings):
                if ('recommendations' in heading.lower() and 'va response to recommendations' not in heading.lower()):  # Case-insensitive search
                    start_index = document_text.lower().find(heading.lower())  # Case-insensitive search
                    if i + 1 < len(headings):
                        end_index = document_text.lower().find(headings[i + 1].lower())  # Case-insensitive search
                    recommendations_section = document_text[start_index:end_index].strip()
                    recommendations_section = recommendations_section.lower()
                    extracted_sections[document_name] = recommendations_section
                    original_texts[document_name] = document_text
                    break

    corpus = pd.DataFrame.from_dict(extracted_sections, orient='index', columns=['recommendations'])
    original_texts_df = pd.DataFrame.from_dict(original_texts, orient='index', columns=['original_text'])

    # Join the 'corpus' DataFrame onto the 'original_texts_df' DataFrame using the index
    corpus = corpus.join(original_texts_df)

    #def remove_words_and_patterns(document, words_to_remove, patterns_to_remove):
        # Split the document into words
        #words = document.split()

        # Clean the words by removing specified words and patterns
        #cleaned_words = [word for word in words if word.lower() not in words_to_remove and not any(re.match(pattern, word) for pattern in patterns_to_remove)]

        # Join the cleaned words to form the cleaned document
        #cleaned_document = " ".join(cleaned_words)
       # return cleaned_document

    # List of words to remove
    words_to_remove = ['veteran','veterans' ,'woman',"women", 'va', 'committee', 'program', 'center', 'study', 'report', 'service', 'within',
                    'include', 'provide', 'ensure', 'develop', 'must', 'need', 'level','department','administration','affairs','veterans benefits administration'
                    ,'acwv']

    # List of patterns to remove (e.g., '2.', '11.', '12.', etc.) with dates excluded as well
    #patterns_to_remove = [r'\d+\.', r'\d+\)',r'\b(?=199\d|20\d{2})\d{4}|january\s\d{1,2}\s|february\s\d{1,2}\s|march\s\d{1,2}\s|april\s\d{1,2}\s|may\s\d{1,2}\s|june\s\d{1,2}\s|july\s\d{1,2}\s|august\s\d{1,2}\s|september\s\d{1,2}\s|october\s\d{1,2}\s|november\s\d{1,2}\s|december\s\d{1,2}\s']

    #for i, row in corpus.iterrows():
        #recommendations = row['recommendations'].lower()
        #original_text = row['original_text'].lower()
        # Remove the specified words and patterns from the recommendations and convert it to lowercase
        #cleaned_document = remove_words_and_patterns(recommendations, words_to_remove, patterns_to_remove)
        #cleaned_document = remove_words_and_patterns(original_text, words_to_remove, patterns_to_remove)

    file_path = "./data/words_dict.csv"

    words_dict = {}

    with open(file_path, 'r', encoding='utf-8', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)  # Use DictReader to read rows as dictionaries

        for row in csv_reader:
            # Assuming the CSV file has 'Title' and 'Meaning' columns
            title = row['Title']
            meaning = row['Meaning']
            words_dict[title] = meaning  # Add the data to the dictionary

    # Convert word dict dataframe to dictionary
    words_dict = {key.lower(): value for key, value in words_dict.items()}

    preprocessed_text = []  # Continue with your code as needed

    def preprocess_text(text):

        # Convert the text to lowercase
        text = text.lower()

        # Remove punctuation
        #my_punctuation = '”!"#$%&()*+,\'''/:;<=>?@[\\]’^_`{|}~“•'
        #text = text.translate(str.maketrans("", "", my_punctuation))

        # Replace hyphens with spaces
        #text = text.replace("-", " ")

        # Tokenize the text into words
        words = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]

        # Lemmatization
        #lemmatizer = WordNetLemmatizer()
        #words = [lemmatizer.lemmatize(word) for word in words]

        # Rejoin the processed words into a single text
        processed_text = " ".join(words)

        # Lowercase the words
        processed_text = processed_text.lower()

        # Replace acronyms with

        return processed_text

    [preprocessed_text.append(preprocess_text(i)) for i in corpus['recommendations']]

    spell_checked_text = []

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")

    if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
        print("Dictionary file not found")

    def spell_check_and_correct(input_text):
        suggestions = sym_spell.lookup_compound(input_text, max_edit_distance=2)
        corrected_text = suggestions[0].term if suggestions else input_text
        return corrected_text

   
    [spell_checked_text.append(spell_check_and_correct(i)) for i in preprocessed_text]

    translated_text = []

    #function to replace acronyms with plain text
    def replace_words(text, acronym_dict):
        words = text.split()
        replaced_words = [acronym_dict.get(word, word) for word in words]
        replaced_text = ' '.join(replaced_words)
        replaced_text = replaced_text.lower()
        return replaced_text
   
    [translated_text.append(replace_words(item, words_dict)) for item in spell_checked_text]
    return corpus,translated_text 

#split text into sentences and add the document year to each sentence
def spacyLayer(text,corpus):
    index_to_year = {}
    
    for i in range(len(corpus)):
        index_to_year[i] = corpus.index[i]
    
    # Create a new list to store sentences with updated indices
    sentences_with_years = []
    
    # Iterate through the sentences and rename the indices
    for index, sentence in enumerate(text):
        year = index_to_year.get(index, None)
    if year is not None:
        sentences_with_years.append(f"{year}: {sentence}")
    return sentences_with_years
    
def append_years(sentences):
    # Initialize an empty list to store sentences with year appended
    sentences_with_years_appended = []
    
    # Iterate through each document in sentences_with_years
    for document in sentences:
        # Split the document into sentence text and year
        year, sentence_text = document.split(": ", 1)
 
    # Parse the sentence using spaCy
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(sentence_text)
    
    # Iterate through each sentence in the document
    for sentence in doc.sents:
        # Append the sentence with year appended
        sentence_with_year = f"{sentence.text} ({year})"
        sentences_with_years_appended.append(sentence_with_year)
        
    filtered_sentences = [sentence for sentence in sentences_with_years_appended if len(sentence) >= 15]
    
    return filtered_sentences
 
#Extracts timestamps for topics over time visulization
def datetime_layer(text):
          
    # Create a list of dictionaries with 'sentence' and 'date' attributes
    sentences_with_dates = []
    
    for sentence in text:
        year_pattern = r'(\d{4}) ACWV Report'
        matches = re.search(year_pattern, sentence)
            
        if matches:
            year = int(matches.group(1))
            date_obj = datetime.date(year=year, month=1, day=1)
            sentences_with_dates.append({'sentence': sentence, 'date': date_obj})
        else:
            sentences_with_dates.append({'sentence': sentence, 'date': None})
        
    timestamps = [item['date'] for item in sentences_with_dates]

    timestamps = pd.to_datetime(timestamps)
    
    return timestamps

    

#define function for BERTopic Modeling of Corpus 
@st.cache_data
def bertopic_model_text(text):

    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-mpnet-base-v2")    

    # Step 2 - Dimensionality Reduction

    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine')

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    # create stopword list
    #create custom stop words vector for this particular corpus
    custom_stop_words = ['veteran','veterans' ,'woman',"women", 'va', 'committee', 'program', 'center', 'study', 'report', 'service', 'within',
                    'include', 'provide', 'ensure', 'develop', 'must', 'need', 'level','department','administration','affairs','veterans benefits administration'
                    ]

    # Instantiate default English stop words
    default_stop_words = set(CountVectorizer(stop_words="english").get_stop_words())

    # Combine default English stop words with custom stop words
    stop_words = list(default_stop_words) + custom_stop_words

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic representations with
    # a `bertopic.representation` model
    representation_model = KeyBERTInspired()

    # All steps together
    topic_model = BERTopic(
    embedding_model=embedding_model,          # Step 1 - Extract embeddings
    umap_model= umap_model,                    # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
    representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
    )
    topics, probs = topic_model.fit_transform(text)
    return topic_model, topics, probs

def get_intertopic_dist_map(topic_model):
    return topic_model.visualize_topics()


#THIS DOESNT WORK
""" def topics_over_time_table(topic_model, timestamps, translated_text):
    # Calculate topics over time
    topics_over_time = topic_model.topics_over_time(docs=translated_text,
                                                timestamps=timestamps,
                                                global_tuning=True,
                                                evolution_tuning=True,
                                                nr_bins=15)
    # Number of rows to display per page
    rows_per_page = 20

    # Calculate the number of pages
    num_pages = len(topics_over_time) // rows_per_page

    # Display the paginated DataFrame
    page_num = st.number_input('Page Number', min_value=1, max_value=num_pages, value=1)
    start_idx = (page_num - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    if start_idx < len(topics_over_time):
        st.table(topics_over_time[start_idx:end_idx])
    else:
        st.write("No more data to display on this page.")

def plot_topics_over_time(topic_model, timestamps, translated_text):
    # Calculate topics over time
    topics_over_time = topic_model.topics_over_time(docs=translated_text,
                                                timestamps=timestamps,
                                                global_tuning=True,
                                                evolution_tuning=True,
                                                nr_bins=15)
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Group topics by their 'Topic' column
    topics_grouped = topics_over_time.groupby('Topic')

    # Plot each topic's frequency over time
    for topic, group in topics_grouped:
        ax.plot(group['Timestamp'], group['frequency'], label=f"Topic {topic}")

    # Customize the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Topics Over Time")
    ax.legend(loc="upper right")

    # Display the plot in Streamlit
    st.pyplot(fig)
 

def show_doc_info(topic_model,translated_text):
    document_info = topic_model.get_document_info(translated_text)
    st.write(document_info)
    """

def prove_success_func(topic_model):
    if topic_model is not None:
        st.write("Topic Model generated successfully.")
        st.write(topic_model)
    else:
        raise ValueError('Topic model not found')
    



 
 
    
