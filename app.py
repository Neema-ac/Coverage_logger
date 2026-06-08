import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF
import base64
from io import BytesIO
import json

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
.highlight {
    background-color: #fff3cd;
    padding: 0.2rem;
    border-radius: 3px;
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
        # Try multiple methods to get title
        title = None
        
        # Method 1: Standard title tag
        if soup.title:
            title = soup.title.string
        
        # Method 2: Open Graph title
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content', '')
        
        # Method 3: H1 tag
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text()
        
        # Clean up title
        if title:
            title = re.sub(r'\s+', ' ', title).strip()
            title = title[:100]  # Limit length
        
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
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 50][:20]
            
            return {
                'text': text[:8000],
                'title': title,
                'key_sentences': key_sentences,
                'url': url
            }
            
        except Exception as e:
            return {
                'text': '',
                'title': f"Error: {url}",
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
        
        # Check for positive/negative indicators
        positive_phrases = [p for p in self.positive_indicators if p in text_lower]
        negative_phrases = [n for n in self.negative_indicators if n in text_lower]
        
        explanation = ""
        key_phrases = []
        
        if sentiment == 'positive':
            explanation = f"The content shows positive sentiment with {confidence:.1%} confidence. "
            if positive_phrases:
                explanation += f"Key positive indicators found: {', '.join(positive_phrases[:3])}. "
                key_phrases.extend(positive_phrases[:3])
            if negative_phrases:
                explanation += f"Some negative indicators present but outweighed by positive content."
        elif sentiment == 'negative':
            explanation = f"The content shows negative sentiment with {confidence:.1%} confidence. "
            if negative_phrases:
                explanation += f"Key negative indicators found: {', '.join(negative_phrases[:3])}. "
                key_phrases.extend(negative_phrases[:3])
        else:
            explanation = f"The content shows neutral sentiment (confidence: {confidence:.1%}). "
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
                    f"• {pillar.replace('_', ' ').title()}: Found keywords {', '.join(matched_keywords[:3])}"
                )
        
        if len(matches) >= 2:
            status = 'YES'
            explanation = f"Strong alignment detected! {len(matches)} core positioning pillars found:\n" + "\n".join(explanation_parts)
        elif len(matches) == 1:
            status = 'PARTIAL'
            explanation = f"Partial alignment detected. Found 1 core positioning pillar:\n" + "\n".join(explanation_parts)
        else:
            status = 'NO'
            explanation = "No alignment with core positioning pillars detected. The content doesn't mention frontier investment, climate finances, or energy access."
        
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
                    'confidence': len(matched_keywords) / len(config['keywords']) * 100
                })
        
        # Sort by confidence
        outcomes.sort(key=lambda x: x['confidence'], reverse=True)
        top_outcomes = outcomes[:3]
        
        if not top_outcomes and matched_pillars and sentiment == 'positive':
            top_outcomes = [{
                'outcome': 'Investor Narrative',
                'keywords': [],
                'description': 'Strengthens investor confidence or fundraising narrative',
                'confidence': 60,
                'explanation': 'High alignment and positive sentiment suggest investor narrative potential.'
            }]
        
        # Generate explanation
        if top_outcomes:
            outcome_names = [o['outcome'] for o in top_outcomes]
            explanation = f"Identified {len(top_outcomes)} potential business outcomes: {', '.join(outcome_names)}. "
            for outcome in top_outcomes:
                if outcome['keywords']:
                    explanation += f"\n• {outcome['outcome']}: Found keywords {', '.join(outcome['keywords'][:3])}"
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
            'key_sentences': page_data['key_sentences'][:5],  # Top 5 sentences for context
            'text_preview': page_data['text'][:500] + "..."
        }
    
    # Sentiment indicators
    positive_indicators = [
        "success", "achievement", "milestone", "growth", "expansion",
        "launch", "breakthrough", "innovation", "leadership", "excellent",
        "sustainable", "positive", "opportunity", "benefit"
    ]
    
    negative_indicators = [
        "criticism", "lawsuit", "controversy", "scandal", "failure",
        "loss", "decline", "risk", "warning", "alert", "problem",
        "challenge", "concern", "issue"
    ]

class PDFReport(FPDF):
    """Custom PDF report generator"""
    
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Company Image Sentiment Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_analysis_section(self, result):
        """Add analysis section for a single URL"""
        # Title
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(102, 126, 234)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"Article: {result['title'][:80]}", 0, 1, 'L', 1)
        self.set_text_color(0, 0, 0)
        
        # URL
        self.set_font('Arial', 'I', 9)
        self.cell(0, 5, f"URL: {result['url']}", 0, 1, 'L')
        self.ln(3)
        
        # Sentiment
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, "Sentiment Analysis", 0, 1, 'L', 1)
        self.set_font('Arial', '', 10)
        
        sentiment_color = ''
        if result['sentiment']['label'] == 'positive':
            sentiment_color = 'Positive'
        elif result['sentiment']['label'] == 'negative':
            sentiment_color = 'Negative'
        else:
            sentiment_color = 'Neutral'
        
        self.cell(0, 6, f"Result: {sentiment_color.upper()} (Confidence: {result['sentiment']['score']:.1%})", 0, 1, 'L')
        self.multi_cell(0, 5, f"Explanation: {result['sentiment']['explanation']}")
        self.ln(3)
        
        # Positioning Alignment
        self.set_font('Arial', 'B', 10)
        self.cell(0, 8, "Positioning Alignment", 0, 1, 'L', 1)
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f"Status: {result['alignment']['status']}", 0, 1, 'L')
        self.multi_cell(0, 5, f"Explanation: {result['alignment']['explanation']}")
        self.ln(3)
        
        # Business Outcomes
        self.set_font('Arial', 'B', 10)
        self.cell(0, 8, "Business Outcomes", 0, 1, 'L', 1)
        self.set_font('Arial', '', 10)
        
        if result['outcomes']['outcomes']:
            outcomes_list = ', '.join([o['outcome'] for o in result['outcomes']['outcomes']])
            self.cell(0, 6, f"Identified: {outcomes_list}", 0, 1, 'L')
        self.multi_cell(0, 5, f"Explanation: {result['outcomes']['explanation']}")
        self.ln(5)
        
        # Key Sentences
        if result.get('key_sentences'):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 8, "Key Excerpts", 0, 1, 'L', 1)
            self.set_font('Arial', '', 9)
            for i, sentence in enumerate(result['key_sentences'][:3], 1):
                self.multi_cell(0, 5, f"{i}. \"{sentence[:200]}...\"")
            self.ln(5)
        
        self.ln(5)

def main():
    # Header
    st.markdown("""
    <div class="big-title">
        <h1>🏢 Company Image Sentiment Analyzer</h1>
        <p>With AI-powered explanations and detailed PDF reports</p>
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
        st.success(f"✅ Analysis complete! {len([r for r in results if not r['error']])} URLs successfully analyzed")
        
        # Display detailed results
        display_detailed_results(results)
        
        # PDF Export option
        st.markdown("---")
        st.markdown("### 📄 Export Report")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📥 Download PDF Report", type="primary", use_container_width=True):
                generate_pdf_report(results)
        
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
        st.metric("✅ Analyzed", len(successful))
    with col2:
        pos_count = sum(1 for r in successful if r['sentiment']['label'] == 'positive')
        st.metric("😊 Positive", pos_count)
    with col3:
        aligned = sum(1 for r in successful if r['alignment']['status'] == 'YES')
        st.metric("🎯 Aligned", aligned)
    with col4:
        avg_confidence = sum(r['sentiment']['score'] for r in successful) / len(successful) if successful else 0
        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
    
    st.markdown("---")
    
    # Detailed results with expanders
    st.markdown("### 📋 Detailed Analysis Results")
    
    for idx, result in enumerate(results, 1):
        with st.expander(f"{idx}. {result['title'][:80]}", expanded=False):
            # Status indicator
            if result['error']:
                st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
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
            
            # Business Outcomes Section
            st.markdown("#### 💼 Business Outcomes")
            if result['outcomes']['outcomes']:
                outcomes_html = "<div class='explanation-box'>"
                outcomes_html += f"<strong>Identified Outcomes:</strong> {', '.join([o['outcome'] for o in result['outcomes']['outcomes']])}<br>"
                outcomes_html += f"<strong>Explanation:</strong> {result['outcomes']['explanation']}<br>"
                
                # Show keyword matches
                for outcome in result['outcomes']['outcomes']:
                    if outcome.get('keywords'):
                        outcomes_html += f"<br><strong>{outcome['outcome']}:</strong> Found keywords: {', '.join(outcome['keywords'][:3])}"
                
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

def generate_pdf_report(results):
    """Generate and download PDF report"""
    try:
        pdf = PDFReport()
        pdf.add_page()
        
        # Add summary section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Executive Summary", 0, 1, 'L')
        pdf.set_font('Arial', '', 11)
        
        successful = [r for r in results if not r.get('error', False)]
        pdf.cell(0, 6, f"Total Articles Analyzed: {len(results)}", 0, 1, 'L')
        pdf.cell(0, 6, f"Successfully Analyzed: {len(successful)}", 0, 1, 'L')
        
        if successful:
            pos_count = sum(1 for r in successful if r['sentiment']['label'] == 'positive')
            aligned_count = sum(1 for r in successful if r['alignment']['status'] == 'YES')
            pdf.cell(0, 6, f"Positive Sentiment: {pos_count}", 0, 1, 'L')
            pdf.cell(0, 6, f"Strong Alignment: {aligned_count}", 0, 1, 'L')
        
        pdf.ln(10)
        
        # Add individual analyses
        for result in results:
            if not result.get('error', False):
                pdf.add_analysis_section(result)
                pdf.add_page()
        
        # Remove last empty page if needed
        if pdf.page_no() > 1 and pdf.get_y() < 50:
            pass
        
        # Save and download
        pdf_output = BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin1'))
        pdf_output.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label="📥 Click to Download PDF Report",
            data=pdf_output,
            file_name=f"sentiment_analysis_report_{timestamp}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        st.success("✅ PDF report generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        st.info("Note: PDF generation requires additional setup. You can still export results as CSV.")

if __name__ == "__main__":
    main()
