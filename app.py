import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from typing import Dict, List, Tuple
import re
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Company Image Sentiment Analyzer",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class CompanyImageAnalyzer:
    """Analyzes website content for sentiment, positioning, and business outcomes."""
    
    def __init__(self):
        @st.cache_resource
        def load_model():
            with st.spinner("Loading AI model (first time takes ~30 seconds)..."):
                return pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
        self.sentiment_analyzer = load_model()
        
        # ========== CUSTOMIZE THESE FOR YOUR COMPANY ==========
        
        # Core positioning pillars
        self.positioning_pillars = {
            "frontier_investment": [
                "frontier market", "emerging market", "high growth", "early stage",
                "venture", "private equity", "innovative finance", "risk capital"
            ],
            "climate_finances": [
                "climate finance", "green bond", "carbon credit", "renewable energy",
                "decarbonization", "net zero", "climate resilience", "esg investing"
            ],
            "energy_access": [
                "energy access", "off-grid", "mini-grid", "clean cooking",
                "electricity access", "solar home system", "last mile energy"
            ]
        }
        
        # Business outcomes
        self.outcome_mapping = {
            "BD Support": ["partnership", "collaboration", "joint venture", "deal", "alliance"],
            "Investor Narrative": ["funding", "investment", "returns", "roi", "valuation", "exit"],
            "Talent & Reputation": ["hiring", "talent", "culture", "award", "recognition"],
            "Partnership": ["partner", "collaboration", "mou", "agreement"],
            "Award/Recognition": ["award", "recognition", "ranked", "top", "winner"]
        }
        
        self.negative_indicators = ["criticism", "lawsuit", "controversy", "scandal", "failure"]
        self.positive_indicators = ["success", "achievement", "milestone", "growth", "launch"]
    
    def fetch_text_from_url(self, url: str) -> str:
        """Fetch and extract text from URL."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = requests.get(url, timeout=12, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
                element.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            return text[:10000]
            
        except Exception as e:
            return ""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text."""
        if not text or len(text.strip()) < 50:
            return {"label": "neutral", "score": 0.5}
        
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        sentiments = []
        
        for chunk in chunks[:3]:
            if len(chunk.strip()) > 10:
                try:
                    result = self.sentiment_analyzer(chunk[:512])[0]
                    sentiments.append(result)
                except:
                    continue
        
        if not sentiments:
            return {"label": "neutral", "score": 0.5}
        
        pos_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        neg_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
        
        text_lower = text.lower()
        neg_matches = sum(1 for kw in self.negative_indicators if kw in text_lower)
        pos_matches = sum(1 for kw in self.positive_indicators if kw in text_lower)
        
        adjustment = (pos_matches - neg_matches) * 0.05
        final_score = ((pos_count - neg_count) / len(sentiments)) + adjustment
        
        if final_score > 0.2:
            return {"label": "positive", "score": round(abs(final_score), 3)}
        elif final_score < -0.2:
            return {"label": "negative", "score": round(abs(final_score), 3)}
        else:
            return {"label": "neutral", "score": 0.5}
    
    def check_positioning_alignment(self, text: str) -> Tuple[str, List[str]]:
        """Check alignment with core positioning."""
        if not text:
            return "no", []
        
        text_lower = text.lower()
        pillars_matched = []
        
        for pillar, keywords in self.positioning_pillars.items():
            for kw in keywords:
                if kw in text_lower:
                    pillars_matched.append(pillar)
                    break
        
        if len(pillars_matched) >= 2:
            return "yes", pillars_matched
        elif len(pillars_matched) == 1:
            return "partial", pillars_matched
        else:
            return "no", []
    
    def determine_business_outcomes(self, text: str, sentiment: str, pillars_matched: List[str]) -> List[str]:
        """Determine business outcomes."""
        outcomes = set()
        text_lower = text.lower()
        
        for outcome, keywords in self.outcome_mapping.items():
            if any(kw in text_lower for kw in keywords):
                if outcome == "Award/Recognition" and sentiment != "positive":
                    continue
                outcomes.add(outcome)
        
        if not outcomes and pillars_matched and sentiment == "positive":
            outcomes.add("Investor Narrative")
        
        if not outcomes:
            outcomes.add("BD Support")
        
        return list(outcomes)[:3]
    
    def analyze_url(self, url: str) -> Dict:
        """Analyze a single URL."""
        text = self.fetch_text_from_url(url)
        
        if not text:
            return {
                "url": url,
                "status": "Failed",
                "sentiment": "N/A",
                "confidence": 0,
                "alignment": "N/A",
                "pillars": "",
                "outcomes": "Could not fetch content",
                "error": True
            }
        
        sentiment = self.analyze_sentiment(text)
        alignment, pillars = self.check_positioning_alignment(text)
        outcomes = self.determine_business_outcomes(text, sentiment["label"], pillars)
        
        return {
            "url": url,
            "status": "Success",
            "sentiment": sentiment['label'].upper(),
            "confidence": sentiment['score'],
            "alignment": alignment.upper(),
            "pillars": ", ".join(pillars) if pillars else "None",
            "outcomes": ", ".join(outcomes),
            "error": False
        }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🏢 Company Image Sentiment Analyzer</h1>
        <p style="color: white; margin: 10px 0 0 0; font-size: 18px;">
            Analyze website content for sentiment, positioning alignment, and business outcomes
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CompanyImageAnalyzer()
    
    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        max_workers = st.slider("Parallel workers", 1, 5, 3)
        st.markdown("---")
        st.markdown("### 🎯 Customization")
        st.info("Edit the `positioning_pillars` and `outcome_mapping` in the code to customize for your company.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter URLs to Analyze")
        
        input_method = st.radio(
            "Choose input method:",
            ["Paste URLs", "Use Examples"],
            horizontal=True
        )
        
        urls = []
        
        if input_method == "Paste URLs":
            url_text = st.text_area(
                "Enter one URL per line:",
                height=200,
                placeholder="https://example.com\nhttps://another-site.com"
            )
            urls = [u.strip() for u in url_text.split('\n') if u.strip()]
        else:
            example_urls = [
                "https://www.climatefinancelab.org",
                "https://www.energyaccess.org",
                "https://www.ifc.org",
                "https://www.worldbank.org/en/topic/climatefinance"
            ]
            urls = example_urls
            st.info(f"Loaded {len(urls)} example URLs")
            for url in urls:
                st.code(url)
        
        analyze_button = st.button("🚀 Analyze URLs", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Batch Analysis</h3>
            <p>Parallel processing for fast results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis execution
    if analyze_button and urls:
        with st.spinner(f"Analyzing {len(urls)} URLs with {max_workers} workers..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(st.session_state.analyzer.analyze_url, url): url for url in urls}
                
                for idx, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(urls))
                    status_text.text(f"Analyzed {idx + 1}/{len(urls)} URLs")
            
            # Store results
            st.session_state.results = results
            
            # Display results
            display_results(results)
            
    elif analyze_button and not urls:
        st.warning("Please enter at least one URL to analyze")
    
    # Display previous results
    elif 'results' in st.session_state and st.session_state.results:
        if st.button("Show Previous Results"):
            display_results(st.session_state.results)

def display_results(results):
    """Display analysis results."""
    successful = [r for r in results if not r.get('error', False)]
    failed = [r for r in results if r.get('error', False)]
    
    # Summary metrics
    st.markdown("---")
    st.markdown("## 📈 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Successful", len(successful))
    with col2:
        st.metric("❌ Failed", len(failed))
    with col3:
        if successful:
            pos_count = sum(1 for r in successful if r['sentiment'] == 'POSITIVE')
            st.metric("😊 Positive", f"{pos_count}/{len(successful)}")
    with col4:
        if successful:
            aligned = sum(1 for r in successful if r['alignment'] == 'YES')
            st.metric("🎯 Aligned", f"{aligned}/{len(successful)}")
    
    # Charts
    if successful:
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = pd.DataFrame([r['sentiment'] for r in successful]).value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=['#27ae60', '#e74c3c', '#f39c12']
            )])
            fig.update_layout(title="Sentiment Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            alignment_counts = pd.DataFrame([r['alignment'] for r in successful]).value_counts()
            fig = go.Figure(data=[go.Bar(
                x=alignment_counts.index,
                y=alignment_counts.values,
                marker_color=['#27ae60', '#f39c12', '#e74c3c']
            )])
            fig.update_layout(title="Positioning Alignment", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.markdown("---")
    st.markdown("## 📋 Detailed Results")
    
    df_display = pd.DataFrame(results)
    display_cols = ['url', 'status', 'sentiment', 'confidence', 'alignment', 'pillars', 'outcomes']
    df_display = df_display[display_cols]
    
    st.dataframe(df_display, use_container_width=True, height=400)
    
    # Export options
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
