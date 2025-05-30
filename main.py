import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import os


# Load environment variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Prompt Template
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Extract news text
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Failed to fetch news from {url} : {e}"

# Summarize news
def summarize_news(url):
    article = extract_news(url)
    if article.startswith("Failed to fetch"):
        return article
    summary = summarize_chain.run(article=article)
    return summary

# Streamlit UI
st.set_page_config(page_title="News Summarizer", layout="centered")
st.title("📰 AI-Powered News Summarizer")

url = st.text_input("Enter a news article URL:")

if st.button("Summarize"):
    if url:
        with st.spinner("Fetching and summarizing the article..."):
            result = summarize_news(url)
        st.subheader("Summary:")
        st.write(result)
    else:
        st.warning("Please enter a valid URL.")

