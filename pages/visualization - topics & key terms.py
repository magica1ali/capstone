import streamlit as st

st.title("BerTopic Visualizations : Topics & Key terms")
st.header("These visualizations provide a view into the individual topic contents and help the user accurately label topics.")

st.header("Topic Information")
st.write("This easily expadable spreadsheet view allows the user to view the contents of each topic, sorted by the topic relevance within the collection of texts. Topics have a name and representations, which is a set of words that best defines the topic. The representations must be interpreted for the topic to be meaningful.")
st.write(st.session_state.fig0) # Topic Information 

st.header("Topic Word Scores Bar charts")
st.write("These bar charts show the top 5 words of the top 10 topics, in order of relevance to the topic. This allows the user to further hone in on what the topics mean.")
st.write(st.session_state.fig4) # visualize the selected terms for a few topics
