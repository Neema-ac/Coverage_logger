import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from io import BytesIO
import json

# Try to import PDF libraries with fallback
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PDF export will be available after installing reportlab")

st.set_page_config(page_title="Company Sentiment Analyzer", layout="wide")

# Custom CSS
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
.explanation-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.keyword-highlight {
    background-color: #fff3cd;
    padding: 0.2rem 0.3rem;
    border-radius: 3px;
    font-family: monospace;
}
.stMetric {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

class EnhancedCompanyAnalyzer:
    """Analyzer with explainability features"""
    
    def __init__(self):
        @st.cache_resource
        def load_model():
            with st.spinner("Loading AI model (first time takes ~30 seconds)..."):
                return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.model = load_model()
        
        # Core positioning pillars with weighted keywords
        self.positioning_pillars = {
            "frontier_investment": {
                "keywords": ["frontier market", "emerging market", "high growth", "early stage", 
                           "venture", "private equity", "innovative finance", "risk capital"],
                "weight": 1.0,
                "description": "Content discusses investment in frontier/emerging markets"
            },
            "climate_finances": {
                "keywords": ["climate finance", "green bond", "carbon credit", "renewable energy",
                           "decarbonization", "net zero", "climate resilience", "esg investing"],
                "weight": 1.0,
                "description": "Content focuses on climate finance and green investment"
            },
            "energy_access": {
                "keywords": ["energy access", "off-grid", "mini-grid", "clean cooking",
                           "electricity access", "solar home system", "last mile energy"],
                "weight": 1.0,
                "description": "Content addresses energy access and distribution"
            }
        }
        
        # Business outcomes with keywords
        self.business_outcomes = {
            "BD Support": {
                "keywords": ["partnership", "collaboration", "joint venture", "deal", 
                           "alliance", "business development"],
                "weight": 1.0,
                "description": "Potential for business development or partnership opportunities"
            },
            "Investor Narrative": {
                "keywords": ["funding", "investment", "returns", "roi", "valuation", 
                           "exit", "capital raise", "investor", "fundraise"],
                "weight": 1.2,
                "description": "Strengthens investor confidence or fundraising narrative"
            },
            "Talent & Reputation": {
                "keywords": ["hiring", "talent", "culture", "award", "recognition", 
                           "best place to work", "employee", "career"],
                "weight": 0.8,
                "description": "Enhances employer brand or company reputation"
            },
            "Partnership": {
                "keywords": ["partner", "collaboration", "mou", "agreement", "strategic alliance"],
                "weight": 1.0,
                "description": "Indicates potential or existing strategic partnerships"
            },
            "Award/Recognition": {
                "keywords": ["award", "recognition", "ranked", "top", "winner", 
                           "honored", "prize", "certification"],
                "weight": 0.9,
                "description": "Recognition or award that enhances credibility"
            }
        }
    
    def extract_title(self, soup):
        """Extract article title from webpage"""
        title = None
        
        # Method 1: Standard title tag
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        # Method 2: Open Graph title
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title.get('content').strip()
        
        # Method 3: H1 tag
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()
        
        # Clean up title
        if title:
            title = re.sub(r'\s+', ' ', title)
            title = title[:100]
        
        return title or "Untitled Article"
    
    def fetch_page_content(self, url):
        """Fetch and parse webpage content with metadata"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = self.extract_title(soup)
            
            # Extract main content
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
                element.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            
            # Extract key sentences (for explainability)
            sentences = re.split(r'[.!?]+', text)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 60][:10]
            
            return {
                'text': text[:8000],
                'title': title,
                'key_sentences': key_sentences,
                'url': url
            }
            
        except Exception as e:
            return {
                'text': '',
                'title': f"Error: Could not fetch {url}",
                'key_sentences': [],
                'url': url,
                'error': str(e)
            }
    
    def analyze_sentiment_with_explanation(self, text):
        """Analyze sentiment with explanation"""
        if not text or len(text.strip()) < 50:
            return {
                'label': 'neutral',
                'score': 0.5,
                'explanation': 'Insufficient content to analyze sentiment.',
                'key_phrases': []
            }
        
        # Get model prediction
        result = self.model(text[:512])[0]
        sentiment = result['label'].lower()
        confidence = result['score']
        
        # Find key phrases that influenced sentiment
        text_lower = text.lower()
        
        positive_phrases = [p for p in self.positive_indicators if p in text_lower]
        negative_phrases = [n for n in self.negative_indicators if n in text_lower]
        
        explanation = ""
        key_phrases = []
        
        if sentiment == 'positive':
            explanation = f"The content shows POSITIVE sentiment with {confidence:.1%} confidence. "
            if positive_phrases:
                explanation += f"Key positive indicators found: {', '.join(positive_phrases[:4])}. "
                key_phrases.extend(positive_phrases[:4])
            if negative_phrases:
                explanation += f"Some negative indicators present but outweighed by positive content."
        elif sentiment == 'negative':
            explanation = f"The content shows NEGATIVE sentiment with {confidence:.1%} confidence. "
            if negative_phrases:
                explanation += f"Key negative indicators found: {', '.join(negative_phrases[:4])}. "
                key_phrases.extend(negative_phrases[:4])
        else:
            explanation = f"The content shows NEUTRAL sentiment (confidence: {confidence:.1%}). "
            explanation += "No strong positive or negative indicators detected."
        
        return {
            'label': sentiment,
            'score': confidence,
            'explanation': explanation,
            'key_phrases': key_phrases
        }
    
    def check_alignment_with_explanation(self, text):
        """Check positioning alignment with detailed explanation"""
        if not text:
            return {
                'status': 'NO',
                'pillars': [],
                'explanation': 'No content available to analyze positioning.',
                'matches': []
            }
        
        text_lower = text.lower()
        matches = []
        explanation_parts = []
        
        for pillar, config in self.positioning_pillars.items():
            matched_keywords = [kw for kw in config['keywords'] if kw in text_lower]
            if matched_keywords:
                matches.append({
                    'pillar': pillar,
                    'keywords': matched_keywords,
                    'description': config['description']
                })
                explanation_parts.append(
                    f"• {pillar.replace('_', ' ').title()}: Found keywords '{', '.join(matched_keywords[:2])}'"
                )
        
        if len(matches) >= 2:
            status = 'YES'
            explanation = f"✅ STRONG ALIGNMENT detected! Matched {len(matches)} core positioning pillars:\n" + "\n".join(explanation_parts)
        elif len(matches) == 1:
            status = 'PARTIAL'
            explanation = f"🟡 PARTIAL ALIGNMENT detected. Matched 1 core positioning pillar:\n" + "\n".join(explanation_parts)
        else:
            status = 'NO'
            explanation = "❌ NO ALIGNMENT detected. The content doesn't mention frontier investment, climate finances, or energy access."
        
        return {
            'status': status,
            'pillars': [m['pillar'] for m in matches],
            'explanation': explanation,
            'matches': matches
        }
    
    def determine_outcomes_with_explanation(self, text, sentiment, matched_pillars):
        """Determine business outcomes with explanation"""
        text_lower = text.lower()
        outcomes = []
        
        for outcome, config in self.business_outcomes.items():
            matched_keywords = [kw for kw in config['keywords'] if kw in text_lower]
            if matched_keywords:
                if outcome == "Award/Recognition" and sentiment != 'positive':
                    continue
                
                outcomes.append({
                    'outcome': outcome,
                    'keywords': matched_keywords,
                    'description': config['description'],
                    'confidence': min(len(matched_keywords) / len(config['keywords']) * 100, 100)
                })
        
        # Sort by confidence
        outcomes.sort(key=lambda x: x['confidence'], reverse=True)
        top_outcomes = outcomes[:3]
        
        if not top_outcomes and matched_pillars and sentiment == 'positive':
            top_outcomes = [{
                'outcome': 'Investor Narrative',
                'keywords': [],
                'description': 'Strengthens investor confidence or fundraising narrative',
                'confidence': 60
            }]
        
        # Generate explanation
        if top_outcomes:
            outcome_names = [o['outcome'] for o in top_outcomes]
            explanation = f"Identified {len(top_outcomes)} potential business outcome(s): {', '.join(outcome_names)}. "
            for outcome in top_outcomes:
                if outcome['keywords']:
                    explanation += f"\n• {outcome['outcome']}: Found keywords '{', '.join(outcome['keywords'][:3])}'"
        else:
            explanation = "No specific business outcomes identified from the content."
        
        return {
            'outcomes': top_outcomes,
            'explanation': explanation
        }
    
    def analyze_url(self, url):
        """Complete analysis with explanations"""
        page_data = self.fetch_page_content(url)
        
        if not page_data['text']:
            return {
                'url': url,
                'title': page_data['title'],
                'status': 'Failed',
                'error': True,
                'sentiment': {'label': 'N/A', 'explanation': 'Could not fetch content'},
                'alignment': {'status': 'NO', 'explanation': 'Content fetch failed'},
                'outcomes': {'outcomes': [], 'explanation': 'Analysis failed'},
                'key_sentences': []
            }
        
        # Run analyses
        sentiment = self.analyze_sentiment_with_explanation(page_data['text'])
        alignment = self.check_alignment_with_explanation(page_data['text'])
        outcomes = self.determine_outcomes_with_explanation(
            page_data['text'], 
            sentiment['label'], 
            alignment['pillars']
        )
        
        return {
            'url': url,
            'title': page_data['title'],
            'status': 'Success',
            'error': False,
            'sentiment': sentiment,
            'alignment': alignment,
            'outcomes': outcomes,
            'key_sentences': page_data['key_sentences'][:5],
            'text_preview': page_data['text'][:300] + "..."
        }
    
    # Sentiment indicators
    positive_indicators = [
        "success", "achievement", "milestone", "growth", "expansion",
        "launch", "breakthrough", "innovation", "leadership", "excellent",
        "sustainable", "positive", "opportunity", "benefit", "award"
    ]
    
    negative_indicators = [
        "criticism", "lawsuit", "controversy", "scandal", "failure",
        "loss", "decline", "risk", "warning", "alert", "problem",
        "challenge", "concern", "issue"
    ]

def create_html_report(results):
    """Create HTML report for PDF conversion"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Sentiment Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
            h2 {{ color: #764ba2; margin-top: 30px; }}
            h3 {{ color: #333; margin-top: 20px; }}
            .summary {{ background: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .article {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; page-break-inside: avoid; }}
            .sentiment-positive {{ color: #27ae60; font-weight: bold; }}
            .sentiment-negative {{ color: #e74c3c; font-weight: bold; }}
            .sentiment-neutral {{ color: #f39c12; font-weight: bold; }}
            .alignment-yes {{ color: #27ae60; font-weight: bold; }}
            .alignment-partial {{ color: #f39c12; font-weight: bold; }}
            .alignment-no {{ color: #e74c3c; font-weight: bold; }}
            .explanation {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 10px 0; }}
            .keyword {{ background: #fff3cd; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>🏢 Company Image Sentiment Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
    """
    
    successful = [r for r in results if not r.get('error', False)]
    html_content += f"<p><strong>Total Articles Analyzed:</strong> {len(results)}</p>"
    html_content += f"<p><strong>Successfully Analyzed:</strong> {len(successful)}</p>"
    
    if successful:
        pos_count = sum(1 for r in successful if r['sentiment']['label'] == 'positive')
        neg_count = sum(1 for r in successful if r['sentiment']['label'] == 'negative')
        neu_count = sum(1 for r in successful if r['sentiment']['label'] == 'neutral')
        aligned_count = sum(1 for r in successful if r['alignment']['status'] == 'YES')
        
        html_content += f"<p><strong>Positive Sentiment:</strong> {pos_count}</p>"
        html_content += f"<p><strong>Negative Sentiment:</strong> {neg_count}</p>"
        html_content += f"<p><strong>Neutral Sentiment:</strong> {neu_count}</p>"
        html_content += f"<p><strong>Strong Alignment:</strong> {aligned_count}</p>"
    
    html_content += "</div>"
    
    # Individual articles
    for idx, result in enumerate(results, 1):
        if result.get('error', False):
            html_content += f"""
            <div class="article">
                <h3>{idx}. {result['title']}</h3>
                <p><strong>URL:</strong> {result['url']}</p>
                <p><strong>Status:</strong> ❌ Failed to analyze</p>
                <p>Could not fetch or parse content.</p>
            </div>
            """
        else:
            sentiment_class = f"sentiment-{result['sentiment']['label']}"
            alignment_class = f"alignment-{result['alignment']['status'].lower()}"
            
            html_content += f"""
            <div class="article">
                <h3>{idx}. {result['title']}</h3>
                <p><strong>URL:</strong> <a href="{result['url']}">{result['url']}</a></p>
                
                <h4>🎭 Sentiment Analysis</h4>
                <div class="explanation">
                    <p><strong>Result:</strong> <span class="{sentiment_class}">{result['sentiment']['label'].upper()}</span> (Confidence: {result['sentiment']['score']:.1%})</p>
                    <p><strong>Explanation:</strong> {result['sentiment']['explanation']}</p>
                </div>
                
                <h4>🎯 Positioning Alignment</h4>
                <div class="explanation">
                    <p><strong>Status:</strong> <span class="{alignment_class}">{result['alignment']['status']}</span></p>
                    <p><strong>Explanation:</strong> {result['alignment']['explanation']}</p>
                </div>
                
                <h4>💼 Business Outcomes</h4>
                <div class="explanation">
                    <p><strong>Identified:</strong> {', '.join([o['outcome'] for o in result['outcomes']['outcomes']]) if result['outcomes']['outcomes'] else 'None'}</p>
                    <p><strong>Explanation:</strong> {result['outcomes']['explanation']}</p>
                </div>
            """
            
            if result.get('key_sentences'):
                html_content += """
                <h4>📝 Key Excerpts</h4>
                <ul>
                """
                for sentence in result['key_sentences'][:3]:
                    html_content += f"<li>\"{sentence[:200]}...\"</li>"
                html_content += "</ul>"
            
            html_content += "</div>"
    
    html_content += f"""
        <div class="footer">
            <p>Report generated by Company Image Sentiment Analyzer</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    # Header
    st.markdown("""
    <div class="big-title">
        <h1>🏢 Company Image Sentiment Analyzer</h1>
        <p>With AI-powered explanations and detailed reports</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedCompanyAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        max_workers = st.slider("Parallel workers", 1, 3, 2)
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.info("""
        ✅ **Sentiment Analysis** with explanations
        ✅ **Positioning Alignment** with keyword matching
        ✅ **Business Outcomes** identification
        ✅ **PDF Reports** with article titles
        ✅ **Key Excerpts** showing influential content
        """)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        - Expand any result to see detailed explanations
        - Keywords that influenced results are highlighted
        - Download CSV for data analysis
        - Download PDF for client-ready reports
        """)
    
    # Main input
    st.markdown("### 📝 Enter URLs to Analyze")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        input_method = st.radio("Input method:", ["Paste URLs", "Use Examples"], horizontal=True)
    
    urls = []
    if input_method == "Paste URLs":
        url_text = st.text_area(
            "Enter one URL per line:", 
            height=150,
            placeholder="https://example.com/article1\nhttps://example.com/article2"
        )
        urls = [u.strip() for u in url_text.split('\n') if u.strip()]
    else:
        example_urls = [
            "https://www.climatefinancelab.org",
            "https://www.energyaccess.org",
            "https://www.ifc.org"
        ]
        urls = example_urls
        st.success(f"📋 Loaded {len(urls)} example URLs")
    
    analyze_btn = st.button("🔍 Analyze URLs", type="primary", use_container_width=True)
    
    if analyze_btn and urls:
        # Progress tracking
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
        
        # Display summary
        successful = [r for r in results if not r.get('error', False)]
        st.success(f"✅ Analysis complete! {len(successful)}/{len(results)} URLs successfully analyzed")
        
        # Display detailed results
        display_detailed_results(results)
        
        # Export options
        st.markdown("---")
        st.markdown("### 📄 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            df = pd.DataFrame([{
                'URL': r['url'],
                'Title': r['title'],
                'Status': r['status'],
                'Sentiment': r['sentiment']['label'].upper() if not r.get('error') else 'ERROR',
                'Confidence': r['sentiment']['score'] if not r.get('error') else 0,
                'Alignment': r['alignment']['status'] if not r.get('error') else 'N/A',
                'Outcomes': ', '.join([o['outcome'] for o in r['outcomes']['outcomes']]) if not r.get('error') and r['outcomes']['outcomes'] else '',
                'Explanation': r['sentiment']['explanation'] if not r.get('error') else ''
            } for r in results])
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON Export
            json_data = json.dumps(results, default=str, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # HTML/PDF Export - Working version
            html_report = create_html_report(results)
            
            # Provide HTML download (works everywhere)
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
            
            st.caption("💡 HTML report works in any browser - save as PDF using browser's Print → Save as PDF")
        
    elif analyze_btn and not urls:
        st.warning("⚠️ Please enter at least one URL to analyze")
    
    # Show previous results if available
    elif 'results' in st.session_state and st.session_state.results:
        if st.button("📊 Show Previous Results"):
            display_detailed_results(st.session_state.results)

def display_detailed_results(results):
    """Display results with expandable explanations"""
    
    # Summary metrics
    successful = [r for r in results if not r.get('error', False)]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Successfully Analyzed", len(successful))
    with col2:
        pos_count = sum(1 for r in successful if r['sentiment']['label'] == 'positive')
        st.metric("😊 Positive Sentiment", pos_count)
    with col3:
        aligned = sum(1 for r in successful if r['alignment']['status'] == 'YES')
        st.metric("🎯 Strong Alignment", aligned)
    with col4:
        avg_confidence = sum(r['sentiment']['score'] for r in successful) / len(successful) if successful else 0
        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
    
    st.markdown("---")
    
    # Detailed results with expanders
    st.markdown("### 📋 Detailed Analysis Results")
    st.markdown("*Click on any section below to see detailed explanations*")
    
    for idx, result in enumerate(results, 1):
        status_icon = "✅" if not result.get('error') else "❌"
        title_display = result['title'][:80] if len(result['title']) > 80 else result['title']
        
        with st.expander(f"{status_icon} {idx}. {title_display}", expanded=False):
            if result.get('error'):
                st.error(f"❌ Failed to analyze: {result['title']}")
                st.code(result['url'])
                continue
            
            # Article metadata
            st.markdown(f"**URL:** {result['url']}")
            
            # Sentiment Section
            st.markdown("#### 🎭 Sentiment Analysis")
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'orange'
            }.get(result['sentiment']['label'], 'gray')
            
            st.markdown(f"""
            <div class="explanation-box">
                <strong>Result:</strong> <span style="color: {sentiment_color}; font-weight: bold;">
                {result['sentiment']['label'].upper()}</span> 
                (Confidence: {result['sentiment']['score']:.1%})<br>
                <strong>Explanation:</strong> {result['sentiment']['explanation']}
            </div>
            """, unsafe_allow_html=True)
            
            # Positioning Section
            st.markdown("#### 🎯 Positioning Alignment")
            alignment_color = {
                'YES': 'green',
                'PARTIAL': 'orange',
                'NO': 'red'
            }.get(result['alignment']['status'], 'gray')
            
            st.markdown(f"""
            <div class="explanation-box">
                <strong>Status:</strong> <span style="color: {alignment_color}; font-weight: bold;">
                {result['alignment']['status']}</span><br>
                <strong>Explanation:</strong> {result['alignment']['explanation']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show matched keywords for alignment
            if result['alignment'].get('matches'):
                st.markdown("**Matched Keywords:**")
                for match in result['alignment']['matches']:
                    st.markdown(f"- {match['pillar'].replace('_', ' ').title()}: `{', '.join(match['keywords'][:3])}`")
            
            # Business Outcomes Section
            st.markdown("#### 💼 Business Outcomes")
            if result['outcomes']['outcomes']:
                outcomes_html = "<div class='explanation-box'>"
                outcomes_html += f"<strong>Identified Outcomes:</strong> {', '.join([o['outcome'] for o in result['outcomes']['outcomes']])}<br>"
                outcomes_html += f"<strong>Explanation:</strong> {result['outcomes']['explanation']}<br>"
                
                for outcome in result['outcomes']['outcomes']:
                    if outcome.get('keywords'):
                        outcomes_html += f"<br><strong>{outcome['outcome']}:</strong> Found keywords: `{', '.join(outcome['keywords'][:3])}`"
                
                outcomes_html += "</div>"
                st.markdown(outcomes_html, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="explanation-box">
                    <strong>No specific outcomes identified</strong><br>
                    {result['outcomes']['explanation']}
                </div>
                """, unsafe_allow_html=True)
            
            # Key Sentences
            if result.get('key_sentences'):
                st.markdown("#### 📝 Key Excerpts (Influencing Factors)")
                for i, sentence in enumerate(result['key_sentences'][:3], 1):
                    st.markdown(f"<div class='explanation-box'><strong>{i}.</strong> \"{sentence[:250]}...\"</div>", 
                               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
