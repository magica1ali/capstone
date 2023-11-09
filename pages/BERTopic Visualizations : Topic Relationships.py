import streamlit as st

st.title("BERTopic Visualizations : Topic Relationships")
st.header("These charts show how the topics relate to one another")

st.header("Topic Hierarchy Chart")
st.write("This chart highlights different clusters of topics (the colored spaces near the topic names). This can help to narrow down topics or identify duplicative topic groups")
st.write(st.session_state.fig1) # Hierarchy Chart

st.header("Intertopic Distance Map")
st.write("This chart shows us how topics relate to one another on a two-dimensional plane. topics that are closer together will have more semantic similarity. The clusters of topics may yield insight into the broader themes the topics share.")
st.write(st.session_state.fig2) # Intertopic Distance 

st.header("Topic Heatmap")
st.write("This chart shows similarity between topics on a one-to-one basis. Darker shaded squares represent an increased level of similarity between the topics on that point of the X and Y axis")
st.write(st.session_state.fig3) # Topic similarity