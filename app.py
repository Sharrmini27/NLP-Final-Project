import streamlit as st
from transformers import pipeline
from newspaper import Article
import pandas as pd

# 1. Page Configuration for a professional look
st.set_page_config(page_title="NLP News Summarizer", page_icon="📝")

# 2. Load the NLP Model (Summarization Pipeline)
# We use 'distilbart' because it is fast and accurate for student projects
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# 3. User Interface (Sidebar and Main Title)
st.title("🤖 NLP Final Project: News Summarizer")
st.markdown("This application uses **Natural Language Processing (Transformers)** to summarize long news articles.")

# Input: News URL
url = st.text_input("Paste a news article URL here (e.g., BBC, CNN, etc.):")

if st.button("Summarize Article"):
    if url:
        try:
            with st.spinner('Extracting text and generating summary...'):
                # 4. Scrape the Article
                article = Article(url)
                article.download()
                article.parse()
                
                # 5. Run the NLP Model
                # We limit the max_length to 130 words for a concise summary
                summary = summarizer(article.text, max_length=130, min_length=30, do_sample=False)
                
                # 6. Display Results (This helps with the 'Results and Analysis' rubric)
                st.subheader(f"Article Title: {article.title}")
                
                st.write("### AI Summary:")
                st.success(summary[0]['summary_text'])
                
                # Data for Analysis Section of your report
                st.info(f"Original Length: {len(article.text.split())} words | Summary Length: {len(summary[0]['summary_text'].split())} words")
                
        except Exception as e:
            st.error(f"Error: {e}. Please make sure the URL is valid.")
    else:
        st.warning("Please enter a URL first.")

# Footer for your assignment
st.sidebar.markdown("---")
st.sidebar.write("**Course:** JIE43303 NLP")
st.sidebar.write("**Project:** News Summarizer App")
