# Company Image Sentiment Analyzer
# ---------------------------------
# Install:
#   pip install streamlit trafilatura transformers torch beautifulsoup4 requests pandas reportlab
#
# Run:
#   streamlit run company_sentiment_analyzer.py
#
# Fixes applied (v2):
#   1. Confidence threshold — results below 65% flagged as "Needs Review" in UI, CSV, and HTML.
#   2. Expanded keyword lists — Energy Access, Climate Finance, Frontier Investment pillars
#      now cover C&I solar, energy infrastructure, sub-Saharan, battery storage, etc.
#   3. CB mention gate — Business Outcomes only tagged when CrossBoundary is mentioned
#      in the article; sector/industry articles no longer pick up false-positive outcomes.
#   4. Corrupted content detection — binary/garbled extraction (>25% non-ASCII) voids the
#      result instead of silently producing a meaningless sentiment score.
#   5. Paywall distinction — FT, WSJ, Bloomberg etc. flagged as "Paywall — manual entry
#      required" rather than a generic fetch failure.

import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import trafilatura

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
.review-box {
    background-color: #fff8e1;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #f39c12;
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

# ------------------------------------------------------------------ #
# Known paywall domains — Fix 5
# ------------------------------------------------------------------ #
PAYWALL_DOMAINS = [
    'ft.com', 'wsj.com', 'bloomberg.com', 'economist.com',
    'nytimes.com', 'washingtonpost.com', 'thetimes.co.uk'
]

# CrossBoundary name variants — Fix 3
CB_NAMES = [
    "crossboundary", "cross boundary", "cb energy",
    "cb advisory", "cb access", "crossboundary energy",
    "crossboundary advisory", "crossboundary group"
]


@st.cache_resource
def load_sentiment_model():
    """Load FinBERT once and cache across reruns."""
    with st.spinner("Loading FinBERT model (first time takes ~30 seconds)..."):
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")


class EnhancedCompanyAnalyzer:
    """Analyzer with explainability features and accuracy fixes."""

    def __init__(self):
        self.model = load_sentiment_model()

        # ------------------------------------------------------------------ #
        # Fix 2 — Expanded positioning pillar keyword lists
        # ------------------------------------------------------------------ #
        self.positioning_pillars = {
            "frontier_investment": {
                "keywords": [
                    "frontier market", "emerging market", "high growth", "early stage",
                    "venture", "private equity", "innovative finance", "risk capital",
                    "sub-saharan", "africa investment", "development finance", "dfi",
                    "blended finance", "impact investing", "frontier investment"
                ],
                "weight": 1.0,
                "description": "Content discusses investment in frontier/emerging markets"
            },
            "climate_finances": {
                "keywords": [
                    "climate finance", "green bond", "carbon credit", "renewable energy",
                    "decarbonization", "decarbonisation", "net zero", "esg investing",
                    "clean energy", "energy transition", "green finance", "climate investment",
                    "solar power", "wind power", "battery storage", "solar and battery",
                    "clean power", "low carbon"
                ],
                "weight": 1.0,
                "description": "Content focuses on climate finance and green investment"
            },
            "energy_access": {
                "keywords": [
                    "energy access", "off-grid", "mini-grid", "clean cooking",
                    "electricity access", "solar home system", "last mile energy",
                    "c&i solar", "commercial and industrial", "energy infrastructure",
                    "power demand", "grid stability", "diesel dependency",
                    "renewable power", "electricity supply", "power generation",
                    "energy solution", "power project", "energy platform"
                ],
                "weight": 1.0,
                "description": "Content addresses energy access and distribution"
            }
        }

        # Business outcomes with keywords
        self.business_outcomes = {
            "BD Support": {
                "keywords": ["joint venture", "business development", "alliance"],
                "weight": 1.0,
                "description": "Potential for business development opportunities"
            },
            "Investor Narrative": {
                "keywords": ["funding", "investment", "returns", "roi", "valuation",
                             "exit", "capital raise", "investor", "fundraise"],
                "weight": 1.2,
                "description": "Strengthens investor confidence or fundraising narrative"
            },
            "Talent & Reputation": {
                "keywords": ["hiring", "talent", "culture", "recognition",
                             "best place to work", "employee", "career"],
                "weight": 0.8,
                "description": "Enhances employer brand or company reputation"
            },
            "Partnership": {
                "keywords": ["partner", "partnership", "collaboration", "mou",
                             "agreement", "strategic alliance"],
                "weight": 1.0,
                "description": "Indicates potential or existing strategic partnerships"
            },
            "Award/Recognition": {
                "keywords": ["award", "ranked", "winner", "honored", "prize", "certification"],
                "weight": 0.9,
                "description": "Recognition or award that enhances credibility"
            }
        }

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def extract_title(self, soup):
        """Extract article title from a BeautifulSoup tree (fallback path)."""
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title.get('content').strip()
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()
        if title:
            title = re.sub(r'\s+', ' ', title)[:100]
        return title or "Untitled Article"

    def fetch_page_content(self, url):
        """Fetch a page and extract the clean ARTICLE BODY using trafilatura,
        falling back to a cleaned BeautifulSoup parse if needed.
        Includes Fix 4 (corrupted content) and Fix 5 (paywall detection)."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Fix 5 — Detect known paywall domains before attempting fetch
            is_paywall = any(domain in url for domain in PAYWALL_DOMAINS)

            # 1) Get raw HTML
            html = None
            try:
                html = trafilatura.fetch_url(url)
            except Exception:
                html = None

            if not html:
                response = requests.get(url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                html = response.text

            title = "Untitled Article"
            text = ""
            corrupted = False

            # 2) Primary: trafilatura clean extraction
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                output_format='json'
            )
            if extracted:
                data = json.loads(extracted)
                text = (data.get('text') or "").strip()
                title = (data.get('title') or "").strip() or title

            # 3) Fallback: BeautifulSoup if trafilatura found little/nothing
            if len(text) < 50:
                soup = BeautifulSoup(html, 'html.parser')
                if title == "Untitled Article":
                    title = self.extract_title(soup)
                for element in soup(['script', 'style', 'nav', 'footer',
                                     'header', 'iframe', 'aside', 'form']):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)

            text = re.sub(r'\s+', ' ', text).strip()

            # Fix 4 — Detect corrupted/binary content (e.g. garbled PDF extraction)
            if len(text) > 0:
                non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
                if non_ascii_ratio > 0.25:
                    corrupted = True
                    text = ""  # Void the result — don't analyze garbage
                    title = f"Error: Corrupted content extracted from {url}"

            title = re.sub(r'\s+', ' ', title)[:100] or "Untitled Article"

            sentences = re.split(r'(?<=[.!?])\s+', text)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 60][:10]

            return {
                'text': text[:8000],
                'title': title,
                'key_sentences': key_sentences,
                'url': url,
                'corrupted': corrupted,
                'is_paywall': is_paywall
            }

        except Exception as e:
            is_paywall = any(domain in url for domain in PAYWALL_DOMAINS)
            return {
                'text': '',
                'title': f"Error: Could not fetch {url}",
                'key_sentences': [],
                'url': url,
                'error': str(e),
                'corrupted': False,
                'is_paywall': is_paywall
            }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunk_text(text, max_chars=1500, max_chunks=6):
        """Split clean text into sentence-aligned chunks (~<512 tokens each)."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) + 1 > max_chars and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur.strip():
            chunks.append(cur.strip())
        return (chunks or [text[:max_chars]])[:max_chunks]

    @staticmethod
    def _match_keywords(text_lower, keywords):
        """Word-boundary regex match."""
        found = []
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                found.append(kw)
        return found

    # ------------------------------------------------------------------ #
    # Sentiment — Fix 1 (confidence threshold)
    # ------------------------------------------------------------------ #
    def analyze_sentiment_with_explanation(self, text):
        """3-class sentiment via FinBERT, averaged over chunks.
        Fix 1: results with confidence < 65% flagged as needs_review."""
        if not text or len(text.strip()) < 50:
            return {
                'label': 'neutral',
                'score': 0.5,
                'needs_review': True,
                'explanation': 'Insufficient content to analyze sentiment.',
                'key_phrases': []
            }

        chunks = self._chunk_text(text)
        agg = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

        for ch in chunks:
            out = self.model(ch, truncation=True, max_length=512, top_k=None)
            scores = out[0] if out and isinstance(out[0], list) else out
            for s in scores:
                lab = s['label'].lower()
                if lab in agg:
                    agg[lab] += float(s['score'])

        n = max(len(chunks), 1)
        for k in agg:
            agg[k] /= n

        sentiment = max(agg, key=agg.get)
        confidence = agg[sentiment]

        # Fix 1 — Flag low-confidence results
        needs_review = confidence < 0.65

        review_note = (
            " ⚠️ LOW CONFIDENCE — recommend manual review before logging."
            if needs_review else ""
        )

        explanation = (
            f"FinBERT classified this as {sentiment.upper()} "
            f"(confidence {confidence:.1%}). Class distribution — "
            f"positive {agg['positive']:.0%}, "
            f"neutral {agg['neutral']:.0%}, "
            f"negative {agg['negative']:.0%}. "
            f"Read over {n} text chunk(s) of the extracted article body."
            f"{review_note}"
        )

        return {
            'label': sentiment,
            'score': confidence,
            'needs_review': needs_review,
            'explanation': explanation,
            'distribution': agg,
            'key_phrases': []
        }

    # ------------------------------------------------------------------ #
    # Positioning — Fix 2 (expanded keywords, already in __init__)
    # ------------------------------------------------------------------ #
    def check_alignment_with_explanation(self, text):
        """Check positioning alignment with detailed explanation."""
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
            matched_keywords = self._match_keywords(text_lower, config['keywords'])
            if matched_keywords:
                matches.append({
                    'pillar': pillar,
                    'keywords': matched_keywords,
                    'description': config['description']
                })
                explanation_parts.append(
                    f"• {pillar.replace('_', ' ').title()}: Found keywords "
                    f"'{', '.join(matched_keywords[:2])}'"
                )

        if len(matches) >= 2:
            status = 'YES'
            explanation = (f"✅ STRONG ALIGNMENT detected! Matched {len(matches)} "
                           f"core positioning pillars:\n" + "\n".join(explanation_parts))
        elif len(matches) == 1:
            status = 'PARTIAL'
            explanation = ("🟡 PARTIAL ALIGNMENT detected. Matched 1 core "
                           "positioning pillar:\n" + "\n".join(explanation_parts))
        else:
            status = 'NO'
            explanation = ("❌ NO ALIGNMENT detected. The content doesn't match "
                           "frontier investment, climate finances, or energy access keywords.")

        return {
            'status': status,
            'pillars': [m['pillar'] for m in matches],
            'explanation': explanation,
            'matches': matches
        }

    # ------------------------------------------------------------------ #
    # Business outcomes — Fix 3 (CB mention gate)
    # ------------------------------------------------------------------ #
    def determine_outcomes_with_explanation(self, text, sentiment, matched_pillars):
        """Determine business outcomes.
        Fix 3: only tag outcomes when CrossBoundary is mentioned in the article."""
        text_lower = text.lower()

        # Fix 3 — CB mention gate
        cb_mentioned = any(name in text_lower for name in CB_NAMES)

        if not cb_mentioned:
            return {
                'outcomes': [],
                'cb_mentioned': False,
                'explanation': (
                    "CrossBoundary not mentioned in this article — "
                    "business outcomes not tagged for sector/industry coverage."
                )
            }

        outcomes = []
        for outcome, config in self.business_outcomes.items():
            matched_keywords = self._match_keywords(text_lower, config['keywords'])
            if matched_keywords:
                if outcome == "Award/Recognition" and sentiment != 'positive':
                    continue
                outcomes.append({
                    'outcome': outcome,
                    'keywords': matched_keywords,
                    'description': config['description'],
                    'confidence': min(len(matched_keywords) / len(config['keywords']) * 100, 100)
                })

        outcomes.sort(key=lambda x: x['confidence'], reverse=True)
        top_outcomes = outcomes[:3]

        if not top_outcomes and matched_pillars and sentiment == 'positive':
            top_outcomes = [{
                'outcome': 'Investor Narrative',
                'keywords': [],
                'description': 'Strengthens investor confidence or fundraising narrative',
                'confidence': 60
            }]

        if top_outcomes:
            outcome_names = [o['outcome'] for o in top_outcomes]
            explanation = (f"Identified {len(top_outcomes)} potential business outcome(s): "
                           f"{', '.join(outcome_names)}. ")
            for outcome in top_outcomes:
                if outcome['keywords']:
                    explanation += (f"\n• {outcome['outcome']}: Found keywords "
                                    f"'{', '.join(outcome['keywords'][:3])}'")
        else:
            explanation = "CrossBoundary mentioned but no specific outcomes matched."

        return {
            'outcomes': top_outcomes,
            'cb_mentioned': True,
            'explanation': explanation
        }

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def analyze_url(self, url):
        """Complete analysis pipeline with all fixes applied."""
        page_data = self.fetch_page_content(url)

        # Fix 5 — Distinguish paywall from generic failure
        if page_data.get('is_paywall') and not page_data['text']:
            return {
                'url': url,
                'title': page_data['title'],
                'status': 'Paywall',
                'error': True,
                'paywall': True,
                'corrupted': False,
                'sentiment': {'label': 'neutral', 'score': 0.0,
                              'needs_review': True,
                              'explanation': 'Paywalled article — manual entry required'},
                'alignment': {'status': 'NO', 'explanation': 'Paywalled — content unavailable',
                              'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False,
                             'explanation': 'Paywalled — manual entry required'},
                'key_sentences': []
            }

        # Fix 4 — Corrupted content
        if page_data.get('corrupted') or not page_data['text']:
            status = 'Corrupted' if page_data.get('corrupted') else 'Failed'
            explanation = (
                'Corrupted binary content detected — result voided. Re-run or log manually.'
                if page_data.get('corrupted')
                else 'Could not fetch content.'
            )
            return {
                'url': url,
                'title': page_data['title'],
                'status': status,
                'error': True,
                'paywall': False,
                'corrupted': page_data.get('corrupted', False),
                'sentiment': {'label': 'neutral', 'score': 0.0,
                              'needs_review': True, 'explanation': explanation},
                'alignment': {'status': 'NO', 'explanation': explanation, 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False, 'explanation': explanation},
                'key_sentences': []
            }

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
            'paywall': False,
            'corrupted': False,
            'sentiment': sentiment,
            'alignment': alignment,
            'outcomes': outcomes,
            'key_sentences': page_data['key_sentences'][:5],
            'text_preview': page_data['text'][:300] + "..."
        }


# ------------------------------------------------------------------ #
# HTML Report
# ------------------------------------------------------------------ #
def create_html_report(results):
    """Create HTML report with all fix indicators surfaced."""
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
            .needs-review {{ background: #fff8e1; padding: 15px; border-left: 4px solid #f39c12; margin: 10px 0; }}
            .error-paywall {{ background: #fff3f3; padding: 15px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
            .keyword {{ background: #fff3cd; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            .badge-review {{ background: #f39c12; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; }}
            .badge-paywall {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; }}
            .badge-cb {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; }}
        </style>
    </head>
    <body>
        <h1>🏢 Company Image Sentiment Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><em>v2 — includes confidence flagging, expanded keywords, CB mention gate, corrupted-content voiding, and paywall detection.</em></p>

        <div class="summary">
            <h2>Executive Summary</h2>
    """

    successful = [r for r in results if not r.get('error', False)]
    needs_review = [r for r in successful if r['sentiment'].get('needs_review', False)]
    paywalled = [r for r in results if r.get('paywall', False)]
    corrupted = [r for r in results if r.get('corrupted', False)]
    cb_articles = [r for r in successful if r['outcomes'].get('cb_mentioned', False)]

    html_content += f"<p><strong>Total Articles:</strong> {len(results)}</p>"
    html_content += f"<p><strong>Successfully Analyzed:</strong> {len(successful)}</p>"
    html_content += f"<p><strong>⚠️ Needs Manual Review (low confidence):</strong> {len(needs_review)}</p>"
    html_content += f"<p><strong>🔒 Paywalled (manual entry required):</strong> {len(paywalled)}</p>"
    html_content += f"<p><strong>💥 Corrupted / Failed:</strong> {len(corrupted) + len([r for r in results if r.get('status') == 'Failed'])}</p>"
    html_content += f"<p><strong>CB-Mentioned Articles:</strong> {len(cb_articles)}</p>"

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

    for idx, result in enumerate(results, 1):
        # Fix 5 — Paywall
        if result.get('paywall'):
            html_content += f"""
            <div class="article">
                <h3>{idx}. {result['title']} <span class="badge-paywall">PAYWALL</span></h3>
                <p><strong>URL:</strong> {result['url']}</p>
                <div class="error-paywall">
                    <p>🔒 <strong>Paywalled article — manual entry required.</strong></p>
                    <p>Log sentiment and alignment manually after reading the article directly.</p>
                </div>
            </div>
            """
            continue

        # Fix 4 — Corrupted or failed
        if result.get('error', False):
            status_label = "CORRUPTED" if result.get('corrupted') else "FAILED"
            html_content += f"""
            <div class="article">
                <h3>{idx}. {result['title']}</h3>
                <p><strong>URL:</strong> {result['url']}</p>
                <div class="error-paywall">
                    <p>❌ <strong>Status: {status_label}</strong></p>
                    <p>{result['sentiment']['explanation']}</p>
                </div>
            </div>
            """
            continue

        sentiment_class = f"sentiment-{result['sentiment']['label']}"
        alignment_class = f"alignment-{result['alignment']['status'].lower()}"
        needs_review_flag = result['sentiment'].get('needs_review', False)
        cb_flag = result['outcomes'].get('cb_mentioned', False)

        review_badge = '<span class="badge-review">⚠️ NEEDS REVIEW</span>' if needs_review_flag else ''
        cb_badge = '<span class="badge-cb">CB MENTIONED</span>' if cb_flag else ''

        html_content += f"""
        <div class="article">
            <h3>{idx}. {result['title']} {review_badge} {cb_badge}</h3>
            <p><strong>URL:</strong> <a href="{result['url']}">{result['url']}</a></p>

            <h4>🎭 Sentiment Analysis</h4>
            <div class="{'needs-review' if needs_review_flag else 'explanation'}">
                <p><strong>Result:</strong> <span class="{sentiment_class}">{result['sentiment']['label'].upper()}</span>
                (Confidence: {result['sentiment']['score']:.1%})
                {'<strong> ⚠️ Low confidence — manual review recommended before logging.</strong>' if needs_review_flag else ''}</p>
                <p><strong>Explanation:</strong> {result['sentiment']['explanation']}</p>
            </div>

            <h4>🎯 Positioning Alignment</h4>
            <div class="explanation">
                <p><strong>Status:</strong> <span class="{alignment_class}">{result['alignment']['status']}</span></p>
                <p><strong>Explanation:</strong> {result['alignment']['explanation']}</p>
            </div>

            <h4>💼 Business Outcomes</h4>
            <div class="explanation">
                <p><strong>CB Mentioned:</strong> {'✅ Yes' if cb_flag else '❌ No — outcomes not tagged'}</p>
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

    html_content += """
        <div class="footer">
            <p>Report generated by Company Image Sentiment Analyzer v2</p>
            <p>Fixes: confidence threshold · expanded keywords · CB mention gate · corrupted content voiding · paywall detection</p>
        </div>
    </body>
    </html>
    """

    return html_content


# ------------------------------------------------------------------ #
# Streamlit UI
# ------------------------------------------------------------------ #
def main():
    st.markdown("""
    <div class="big-title">
        <h1>🏢 Company Image Sentiment Analyzer</h1>
        <p>FinBERT-powered analysis with clean article extraction</p>
    </div>
    """, unsafe_allow_html=True)

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedCompanyAnalyzer()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        max_workers = st.slider("Parallel workers", 1, 3, 1)
        st.caption("Keep at 1 on Streamlit Community Cloud (1 GB RAM) to avoid out-of-memory crashes.")

        st.markdown("---")
        st.markdown("### 📊 Accuracy Notes")
        st.info("""
        **Confidence threshold:** Results below 65% are flagged ⚠️ for manual review — don't log these automatically.

        **CB mention gate:** Business Outcomes are only tagged when CrossBoundary appears in the article. Sector news won't produce false-positive outcomes.

        **Paywalled sites** (FT, WSJ, Bloomberg) are flagged for manual entry — the tool won't attempt to guess their content.
        """)

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        - Expand any result to see detailed explanations
        - Download CSV for bulk review — filter 'Needs Review = TRUE' first
        - Download HTML for a client-ready report
        """)

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

        st.session_state.results = results

        successful = [r for r in results if not r.get('error', False)]
        st.success(f"✅ Analysis complete! {len(successful)}/{len(results)} URLs successfully analyzed")

        display_detailed_results(results)

        st.markdown("---")
        st.markdown("### 📄 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            df = pd.DataFrame([{
                'URL': r['url'],
                'Title': r['title'],
                'Status': r['status'],
                'Sentiment': r['sentiment']['label'].upper() if not r.get('error') else 'ERROR',
                'Confidence': f"{r['sentiment']['score']:.1%}" if not r.get('error') else 'N/A',
                # Fix 1 — Needs Review column in CSV
                'Needs Review': r['sentiment'].get('needs_review', True) if not r.get('error') else True,
                'Alignment': r['alignment']['status'] if not r.get('error') else 'N/A',
                # Fix 3 — CB Mentioned column in CSV
                'CB Mentioned': r['outcomes'].get('cb_mentioned', False) if not r.get('error') else False,
                'Outcomes': ', '.join([o['outcome'] for o in r['outcomes']['outcomes']]) if not r.get('error') and r['outcomes']['outcomes'] else '',
                # Fix 5 — Paywall flag in CSV
                'Paywall': r.get('paywall', False),
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
            json_data = json.dumps(results, default=str, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col3:
            html_report = create_html_report(results)
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
            st.caption("💡 Open the HTML in any browser, then Print → Save as PDF")

    elif analyze_btn and not urls:
        st.warning("⚠️ Please enter at least one URL to analyze")

    elif 'results' in st.session_state and st.session_state.results:
        if st.button("📊 Show Previous Results"):
            display_detailed_results(st.session_state.results)


def display_detailed_results(results):
    """Display results with expandable explanations and fix indicators."""
    successful = [r for r in results if not r.get('error', False)]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("✅ Analyzed", len(successful))
    with col2:
        pos_count = sum(1 for r in successful if r['sentiment']['label'] == 'positive')
        st.metric("😊 Positive", pos_count)
    with col3:
        aligned = sum(1 for r in successful if r['alignment']['status'] == 'YES')
        st.metric("🎯 Strong Alignment", aligned)
    with col4:
        # Fix 1 — Surface review count in metrics
        review_count = sum(1 for r in successful if r['sentiment'].get('needs_review', False))
        st.metric("⚠️ Needs Review", review_count)
    with col5:
        avg_confidence = sum(r['sentiment']['score'] for r in successful) / len(successful) if successful else 0
        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")

    st.markdown("---")

    # Fix 1 — Review queue at top
    review_articles = [r for r in successful if r['sentiment'].get('needs_review', False)]
    if review_articles:
        with st.expander(f"⚠️ Manual Review Queue ({len(review_articles)} articles need checking)", expanded=True):
            st.markdown("These articles have sentiment confidence below 65% and should be reviewed before logging to the coverage tracker.")
            for r in review_articles:
                st.markdown(
                    f"- **{r['title'][:70]}** — {r['sentiment']['label'].upper()} "
                    f"({r['sentiment']['score']:.1%} confidence) — [link]({r['url']})"
                )

    # Fix 5 — Paywall queue
    paywall_articles = [r for r in results if r.get('paywall', False)]
    if paywall_articles:
        with st.expander(f"🔒 Paywalled Articles ({len(paywall_articles)} — manual entry required)", expanded=False):
            st.markdown("These articles could not be fetched. Log sentiment and alignment manually after reading directly.")
            for r in paywall_articles:
                st.markdown(f"- {r['url']}")

    st.markdown("### 📋 Detailed Analysis Results")
    st.markdown("*Click on any section below to see detailed explanations*")

    for idx, result in enumerate(results, 1):
        needs_review_flag = result['sentiment'].get('needs_review', False)
        paywall_flag = result.get('paywall', False)
        corrupted_flag = result.get('corrupted', False)
        cb_flag = result['outcomes'].get('cb_mentioned', False) if not result.get('error') else False

        # Build expander label with status badges
        status_icon = "✅" if not result.get('error') else ("🔒" if paywall_flag else "❌")
        review_tag = " ⚠️" if needs_review_flag else ""
        cb_tag = " 🏢" if cb_flag else ""
        title_display = result['title'][:70] if len(result['title']) > 70 else result['title']

        with st.expander(f"{status_icon} {idx}. {title_display}{review_tag}{cb_tag}", expanded=False):

            # Fix 5 — Paywall
            if paywall_flag:
                st.error("🔒 Paywalled article — manual entry required")
                st.code(result['url'])
                continue

            # Fix 4 — Corrupted or failed
            if result.get('error'):
                st.error(f"❌ {result['status']}: {result['title']}")
                st.code(result['url'])
                if corrupted_flag:
                    st.warning("💥 Corrupted binary content was detected and voided. Re-run or log manually.")
                continue

            st.markdown(f"**URL:** {result['url']}")

            # Fix 3 — CB mention indicator
            if cb_flag:
                st.success("🏢 CrossBoundary mentioned — business outcomes tagged")
            else:
                st.info("ℹ️ CrossBoundary not mentioned — sector/industry coverage, outcomes not tagged")

            st.markdown("#### 🎭 Sentiment Analysis")
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'orange'
            }.get(result['sentiment']['label'], 'gray')

            box_class = "review-box" if needs_review_flag else "explanation-box"
            review_warning = "<br><strong>⚠️ Low confidence — manually verify before logging to the coverage tracker.</strong>" if needs_review_flag else ""

            st.markdown(f"""
            <div class="{box_class}">
                <strong>Result:</strong> <span style="color: {sentiment_color}; font-weight: bold;">
                {result['sentiment']['label'].upper()}</span>
                (Confidence: {result['sentiment']['score']:.1%}){review_warning}<br>
                <strong>Explanation:</strong> {result['sentiment']['explanation']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🎯 Positioning Alignment")
            alignment_color = {'YES': 'green', 'PARTIAL': 'orange', 'NO': 'red'}.get(
                result['alignment']['status'], 'gray')

            st.markdown(f"""
            <div class="explanation-box">
                <strong>Status:</strong> <span style="color: {alignment_color}; font-weight: bold;">
                {result['alignment']['status']}</span><br>
                <strong>Explanation:</strong> {result['alignment']['explanation']}
            </div>
            """, unsafe_allow_html=True)

            if result['alignment'].get('matches'):
                st.markdown("**Matched Keywords:**")
                for match in result['alignment']['matches']:
                    st.markdown(f"- {match['pillar'].replace('_', ' ').title()}: `{', '.join(match['keywords'][:3])}`")

            st.markdown("#### 💼 Business Outcomes")
            if result['outcomes']['outcomes']:
                outcomes_html = "<div class='explanation-box'>"
                outcomes_html += f"<strong>Identified Outcomes:</strong> {', '.join([o['outcome'] for o in result['outcomes']['outcomes']])}<br>"
                outcomes_html += f"<strong>Explanation:</strong> {result['outcomes']['explanation']}<br>"
                for outcome in result['outcomes']['outcomes']:
                    if outcome.get('keywords'):
                        outcomes_html += f"<br><strong>{outcome['outcome']}:</strong> Keywords: `{', '.join(outcome['keywords'][:3])}`"
                outcomes_html += "</div>"
                st.markdown(outcomes_html, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="explanation-box">
                    {result['outcomes']['explanation']}
                </div>
                """, unsafe_allow_html=True)

            if result.get('key_sentences'):
                st.markdown("#### 📝 Key Excerpts (from extracted article body)")
                for i, sentence in enumerate(result['key_sentences'][:3], 1):
                    st.markdown(
                        f"<div class='explanation-box'><strong>{i}.</strong> \"{sentence[:250]}...\"</div>",
                        unsafe_allow_html=True
                    )


if __name__ == "__main__":
    main()
