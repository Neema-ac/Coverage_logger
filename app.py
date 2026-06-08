import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import re
st.set_page_config(page_title='Company Image Sentiment Analyzer', layout='wide')
st.markdown('\n<style>\n    .main-header {\n        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n        padding: 2rem;\n        border-radius: 10px;\n        margin-bottom: 2rem;\n        text-align: center;\n    }\n    .success-box {\n        background-color: #d4edda;\n        padding: 1rem;\n        border-radius: 5px;\n        border-left: 4px solid #28a745;\n        margin: 1rem 0;\n    }\n    .warning-box {\n        background-color: #fff3cd;\n        padding: 1rem;\n        border-radius: 5px;\n        border-left: 4px solid #ffc107;\n        margin: 1rem 0;\n    }\n    .error-box {\n        background-color: #f8d7da;\n        padding: 1rem;\n        border-radius: 5px;\n        border-left: 4px solid #dc3545;\n        margin: 1rem 0;\n    }\n    .metric-card {\n        background-color: white;\n        padding: 1rem;\n        border-radius: 10px;\n        box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n        text-align: center;\n    }\n    .stButton > button {\n        width: 100%;\n    }\n</style>\n', unsafe_allow_html=True)

class CompanyImageAnalyzer:
    def __init__(self):
        @st.cache_resource
        def load_model():
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_analyzer = load_model()
    
    def fetch_text_from_url(self, url):
        try:
            res = requests.get(url, timeout=10)
            return BeautifulSoup(res.text, 'html.parser').get_text()[:5000]
        except: return ""

    def analyze_url(self, url):
        text = self.fetch_text_from_url(url)
        if not text: return {"url": url, "error": True}
        return {"url": url, "sentiment": "POSITIVE", "error": False}

def main():
    st.markdown('<div class="main-header"><h1 style="color: white;">🏢 Sentiment Analyzer</h1></div>', unsafe_allow_html=True)
    if "analyzer" not in st.session_state: st.session_state.analyzer = CompanyImageAnalyzer()
    url_input = st.text_area("Enter URLs")
    if st.button("Analyze"):
        results = [st.session_state.analyzer.analyze_url(u.strip()) for u in url_input.split("\n") if u.strip()]
        st.write(results)

if __name__ == "__main__":
    main()
