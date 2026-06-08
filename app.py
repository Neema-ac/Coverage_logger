import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Company Sentiment Analyzer", layout="wide")

st.markdown("""
<style>
.big-title {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
</style>
<div class="big-title">
    <h1>🏢 Company Image Sentiment Analyzer</h1>
    <p>Analyze website sentiment, positioning alignment, and business outcomes</p>
</div>
""", unsafe_allow_html=True)

class CompanyAnalyzer:
    def __init__(self):
        @st.cache_resource
        def load_model():
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.model = load_model()
        
        self.positioning_pillars = {
            "frontier_investment": ["frontier market", "emerging market", "venture", "private equity"],
            "climate_finances": ["climate finance", "green bond", "renewable energy", "carbon credit"],
            "energy_access": ["energy access", "off-grid", "solar home", "clean cooking"]
        }
        
        self.business_outcomes = {
            "BD Support": ["partnership", "deal", "collaboration"],
            "Investor Narrative": ["funding", "investment", "returns"],
            "Talent & Reputation": ["hiring", "award", "recognition"],
            "Partnership": ["partner", "alliance", "mou"],
            "Award/Recognition": ["award", "winner", "ranked"]
        }
    
    def fetch_text(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text)
            return text[:5000]
        except:
            return ""
    
    def check_alignment(self, text):
        if not text:
            return "NO", []
        text_lower = text.lower()
        matched = []
        for pillar, keywords in self.positioning_pillars.items():
            if any(kw in text_lower for kw in keywords):
                matched.append(pillar)
        if len(matched) >= 2:
            return "YES", matched
        elif len(matched) == 1:
            return "PARTIAL", matched
        return "NO", matched
    
    def get_outcomes(self, text, sentiment, matched):
        outcomes = set()
        text_lower = text.lower()
        for outcome, keywords in self.business_outcomes.items():
            if any(kw in text_lower for kw in keywords):
                if outcome == "Award/Recognition" and sentiment != "POSITIVE":
                    continue
                outcomes.add(outcome)
        if not outcomes and matched and sentiment == "POSITIVE":
            outcomes.add("Investor Narrative")
        return list(outcomes)[:3] if outcomes else ["BD Support"]
    
    def analyze(self, url):
        text = self.fetch_text(url)
        if not text:
            return {
                "URL": url,
                "Sentiment": "ERROR",
                "Confidence": 0,
                "Alignment": "NO",
                "Pillars": "",
                "Outcomes": "Fetch failed"
            }
        
        result = self.model(text[:512])[0]
        sentiment = result['label']
        confidence = round(result['score'], 3)
        alignment, matched = self.check_alignment(text)
        outcomes = self.get_outcomes(text, sentiment, matched)
        
        return {
            "URL": url,
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Alignment": alignment,
            "Pillars": ", ".join(matched) if matched else "None",
            "Outcomes": ", ".join(outcomes)
        }

# Initialize
if 'analyzer' not in st.session_state:
    with st.spinner("Loading AI model..."):
        st.session_state.analyzer = CompanyAnalyzer()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    max_workers = st.slider("Parallel workers", 1, 3, 2)
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("Analyzes sentiment, positioning alignment, and business outcomes from website content.")

# Main input
st.markdown("### 📝 Enter URLs")
input_method = st.radio("Input method:", ["Paste URLs", "Use Examples"], horizontal=True)

urls = []
if input_method == "Paste URLs":
    url_text = st.text_area("One URL per line:", height=150)
    urls = [u.strip() for u in url_text.split('\n') if u.strip()]
else:
    example_urls = [
        "https://www.climatefinancelab.org",
        "https://www.energyaccess.org",
        "https://www.ifc.org"
    ]
    urls = example_urls
    st.info(f"Loaded {len(urls)} example URLs")

if st.button("🔍 Analyze URLs", type="primary", use_container_width=True):
    if not urls:
        st.warning("Please enter at least one URL")
    else:
        progress = st.progress(0)
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(st.session_state.analyzer.analyze, url): url for url in urls}
            for i, future in enumerate(as_completed(futures)):
                results.append(future.result())
                progress.progress((i + 1) / len(urls))
        
        st.success(f"✅ Analyzed {len(results)} URLs")
        
        # Display results
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv,
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            use_container_width=True
        )
