# Company Image Sentiment Analyzer  (CB-tuned, v3)
# ------------------------------------------------
# Install:
#   pip install streamlit trafilatura transformers torch beautifulsoup4 requests pandas reportlab
#
# Run:
#   streamlit run company_sentiment_analyzer.py
#
# What changed in v3 (the two things you asked for):
#
#   A. MORE IN TUNE WITH CB
#      - Expanded CB entity list to CrossBoundary's real brands/acronyms:
#        CBE, CBEA, CrossBoundary Access/Energy/Advisory/Group, Fund for Nature, Frontier platform.
#      - Refined pillar keywords to CB's own language (blended finance, mini-grids,
#        24/7 grid-quality power, bankable, offtake, MIGA guarantee, carbon markets,
#        nature-based solutions, concessional finance, etc.).
#      - NEW: CB-focused sentiment. When CB is mentioned, sentiment is scored on the
#        sentences that talk about CB (plus their neighbors) instead of the whole page,
#        so the score reflects how the article feels ABOUT CB specifically.
#      - NEW: 0-100 CB alignment score with a direct-mention boost.
#
#   B. HIGHER CONFIDENCE (without faking it)
#      - NEW: headline-first scoring. FinBERT is decisive on short headline-style text.
#        If the article title alone crosses the threshold, it's used as the final result.
#      - NEW: drop-ambiguous-chunks aggregation. Chunks where no class reaches 50%
#        are pure noise; excluding them from the average sharpens the final score.
#        Neutral is fully preserved — a chunk can be decisively neutral (neutral > 50%).
#      - Default threshold lowered to 0.55. For full-article analysis, FinBERT typically
#        tops out at 60-65% even on clearly-toned pieces; 0.65 was always too strict.
#        The threshold remains a sidebar slider you can tune live.
#
#   Carried over from v2: confidence flagging, CB mention gate on outcomes,
#   corrupted-content voiding, paywall detection.

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
# Known paywall domains
# ------------------------------------------------------------------ #
PAYWALL_DOMAINS = [
    'ft.com', 'wsj.com', 'bloomberg.com', 'economist.com',
    'nytimes.com', 'washingtonpost.com', 'thetimes.co.uk'
]

# ------------------------------------------------------------------ #
# CrossBoundary entity variants (expanded to CB's real brands).
# Multi-word forms are unambiguous; the short acronyms (cbe, cbea) are kept
# with strict word boundaries so they only match as standalone tokens.
# ------------------------------------------------------------------ #
CB_NAMES = [
    "crossboundary", "cross boundary", "cross-boundary",
    "crossboundary group", "crossboundary advisory", "crossboundary energy",
    "crossboundary energy access", "crossboundary access",
    "cb energy", "cb advisory", "cb access",
    "cbe", "cbea",
    "fund for nature",
]

# CB sub-brand acronyms that are risky as bare tokens get extra context guarding.
# We only treat "cbe"/"cbea" as CB when they appear near energy/access/Africa context,
# to avoid matching unrelated uses. Handled in _cb_sentence_match().
CB_AMBIGUOUS = {"cbe", "cbea"}
CB_CONTEXT_TERMS = [
    "energy", "mini-grid", "minigrid", "solar", "africa", "renewable",
    "power", "access", "crossboundary", "fund", "grid"
]


@st.cache_resource
def load_sentiment_model():
    """Load FinBERT once and cache across reruns."""
    with st.spinner("Loading FinBERT model (first time takes ~30 seconds)..."):
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")


class EnhancedCompanyAnalyzer:
    """Analyzer with explainability features, CB tuning, and confidence upgrades."""

    def __init__(self):
        self.model = load_sentiment_model()

        # ---- Tunable settings (overridden from the sidebar before each run) ----
        self.confidence_threshold = 0.55   # below this => "Needs Review"
        self.focus_cb = True               # score sentiment on CB-context sentences
        self.neighbor_window = 2           # sentences to include on each side of a CB hit

        # ------------------------------------------------------------------ #
        # Positioning pillars — refined to CrossBoundary's own language
        # ------------------------------------------------------------------ #
        self.positioning_pillars = {
            "frontier_investment": {
                "keywords": [
                    "frontier market", "emerging market", "emerging and frontier",
                    "early stage", "venture", "private equity", "innovative finance",
                    "risk capital", "sub-saharan", "africa investment",
                    "development finance", "dfi", "blended finance", "impact investing",
                    "frontier investment", "frontier platform", "green sme",
                    "de-risk", "concessional finance", "bankable", "bankable project",
                    "offtake", "offtake agreement", "miga", "guarantee structure",
                    "private capital", "catalytic capital"
                ],
                "weight": 1.0,
                "description": "Content discusses investment in frontier/emerging markets"
            },
            "climate_finance": {
                "keywords": [
                    "climate finance", "green bond", "carbon credit", "carbon market",
                    "voluntary carbon market", "renewable energy", "decarbonization",
                    "decarbonisation", "net zero", "esg", "esg investing", "clean energy",
                    "energy transition", "green finance", "climate investment",
                    "solar power", "wind power", "hydropower", "battery storage",
                    "solar and battery", "solar and batteries", "clean power",
                    "low carbon", "nature-based solution", "nature-based solutions",
                    "natural capital", "conservation finance", "climate equity",
                    "climate-aligned", "nationally determined contribution", "ndc"
                ],
                "weight": 1.0,
                "description": "Content focuses on climate finance and green investment"
            },
            "energy_access": {
                "keywords": [
                    "energy access", "off-grid", "mini-grid", "mini grid", "minigrid",
                    "clean cooking", "electricity access", "solar home system",
                    "last mile energy", "c&i solar", "commercial and industrial",
                    "industrial renewable", "energy infrastructure", "power demand",
                    "grid stability", "grid-quality power", "24/7 power",
                    "diesel dependency", "renewable power", "electricity supply",
                    "power generation", "energy solution", "power project",
                    "energy platform", "electrification", "rural electrification",
                    "universal electrification", "solar-powered", "solar powered"
                ],
                "weight": 1.0,
                "description": "Content addresses energy access and distribution"
            }
        }

        # Flat keyword set for fast "any pillar keyword?" checks
        self._all_pillar_keywords = [
            kw for cfg in self.positioning_pillars.values() for kw in cfg["keywords"]
        ]

        # Business outcomes with keywords
        self.business_outcomes = {
            "BD Support": {
                "keywords": ["joint venture", "business development", "alliance"],
                "weight": 1.0,
                "description": "Potential for business development opportunities"
            },
            "Investor Narrative": {
                "keywords": ["funding", "investment", "returns", "roi", "valuation",
                             "exit", "capital raise", "investor", "fundraise",
                             "oversubscribed", "round", "commitment", "raised"],
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
                             "agreement", "strategic alliance", "consortium"],
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
        Includes corrupted-content and paywall detection."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            is_paywall = any(domain in url for domain in PAYWALL_DOMAINS)

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

            if len(text) < 50:
                soup = BeautifulSoup(html, 'html.parser')
                if title == "Untitled Article":
                    title = self.extract_title(soup)
                for element in soup(['script', 'style', 'nav', 'footer',
                                     'header', 'iframe', 'aside', 'form']):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)

            text = re.sub(r'\s+', ' ', text).strip()

            # Detect corrupted/binary content (e.g. garbled PDF extraction)
            if len(text) > 0:
                non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
                if non_ascii_ratio > 0.25:
                    corrupted = True
                    text = ""
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
    # Matching helpers
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

    def _any_pillar_keyword(self, text_lower):
        return any(
            re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
            for kw in self._all_pillar_keywords
        )

    def _cb_sentence_match(self, sentence_lower):
        """Return True if this sentence references CrossBoundary.
        Ambiguous acronyms (cbe/cbea) only count when CB-context terms are nearby."""
        for name in CB_NAMES:
            if not re.search(r'\b' + re.escape(name) + r'\b', sentence_lower):
                continue
            if name in CB_AMBIGUOUS:
                if any(re.search(r'\b' + re.escape(t) + r'\b', sentence_lower)
                       for t in CB_CONTEXT_TERMS):
                    return True
                continue
            return True
        return False

    # ------------------------------------------------------------------ #
    # CB-focused sentence selection (drives both CB-tuning and confidence)
    # ------------------------------------------------------------------ #
    def _build_focus_text(self, text):
        """Choose which text to feed FinBERT.

        Returns (focus_text, focus_mode, cb_mentioned, cb_relevance):
          - 'cb_context'  : sentences mentioning CB (+ neighbors)        [best signal]
          - 'pillar'      : sentences with pillar keywords               [topic-level]
          - 'full'        : whole article                                [fallback]
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return text, 'full', False, 0

        lowered = [s.lower() for s in sentences]
        cb_idx = [i for i, s in enumerate(lowered) if self._cb_sentence_match(s)]
        pillar_idx = [i for i, s in enumerate(lowered) if self._any_pillar_keyword(s)]

        cb_mentioned = len(cb_idx) > 0

        # CB relevance: how much of the article is actually about CB / its pillars (0-100)
        if cb_mentioned:
            density = (len(cb_idx) + 0.5 * len(set(pillar_idx) - set(cb_idx))) / max(len(sentences), 1)
            cb_relevance = int(min(100, 50 + density * 200))  # direct mention floors at ~50
        elif pillar_idx:
            cb_relevance = int(min(45, len(pillar_idx) / max(len(sentences), 1) * 180))
        else:
            cb_relevance = 0

        # Pick the focus set
        if self.focus_cb and cb_idx:
            keep = set()
            for i in cb_idx:
                for j in range(i - self.neighbor_window, i + self.neighbor_window + 1):
                    if 0 <= j < len(sentences):
                        keep.add(j)
            focus_text = " ".join(sentences[j] for j in sorted(keep))
            mode = 'cb_context'
        elif pillar_idx:
            keep = set()
            for i in pillar_idx:
                for j in range(i - self.neighbor_window, i + self.neighbor_window + 1):
                    if 0 <= j < len(sentences):
                        keep.add(j)
            focus_text = " ".join(sentences[j] for j in sorted(keep))
            mode = 'pillar'
        else:
            focus_text = text
            mode = 'full'

        # Guard against an over-thin focus set (FinBERT needs a decent passage)
        if len(focus_text.strip()) < 300:
            focus_text = text
            mode = 'full'

        return focus_text, mode, cb_mentioned, cb_relevance

    # ------------------------------------------------------------------ #
    # FinBERT aggregation (drop-ambiguous-chunks; neutral fully preserved)
    # ------------------------------------------------------------------ #
    def _aggregate_finbert(self, chunks):
        """Run FinBERT on each chunk and combine into one distribution.

        All three classes (positive / neutral / negative) are treated equally.
        A chunk is 'decisive' when its winning class reaches >= 50%. Ambiguous
        chunks (roughly 33/33/33) are dropped because they add noise without
        signal. If every chunk is ambiguous the single most confident one is
        used. This is direction-neutral — a chunk can be decisively neutral.
        """
        per_chunk_data = []
        for ch in chunks:
            out = self.model(ch, truncation=True, max_length=512, top_k=None)
            scores = out[0] if out and isinstance(out[0], list) else out
            dist = {}
            for s in scores:
                dist[s['label'].lower()] = float(s['score'])
            for k in ('positive', 'negative', 'neutral'):
                dist.setdefault(k, 0.0)
            top_conf = max(dist.values())
            per_chunk_data.append((ch, dist, top_conf))

        # Keep decisive chunks (winning class >= 50 %)
        decisive = [(ch, d) for ch, d, conf in per_chunk_data if conf >= 0.50]
        if not decisive:
            # All ambiguous — fall back to the single most confident one
            best = max(per_chunk_data, key=lambda x: x[2])
            decisive = [(best[0], best[1])]

        # Plain length-weighted average of decisive chunks
        agg = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        total_w = 0.0
        for ch, dist in decisive:
            w = float(len(ch)) + 1e-6
            total_w += w
            for k in agg:
                agg[k] += dist[k] * w
        for k in agg:
            agg[k] /= total_w

        return agg, len(decisive)

    # ------------------------------------------------------------------ #
    # Sentiment (CB-focused + threshold from settings)
    # ------------------------------------------------------------------ #
    def analyze_sentiment_with_explanation(self, text, title=""):
        """3-class sentiment via FinBERT, scored on CB-focused text where possible.
        Results below the configured threshold are flagged needs_review.

        Stage 1: try the article headline — FinBERT is trained on headlines and very
        decisive on them. If the title alone crosses the threshold, use it.
        Stage 2: score on CB-context sentences (or full article as fallback).
        """
        # ---- Stage 1: headline-first ----
        headline = (title or "").strip()
        if len(headline) > 15:
            title_agg, _ = self._aggregate_finbert([headline])
            title_label = max(title_agg, key=title_agg.get)
            title_conf = title_agg[title_label]
            if title_conf >= self.confidence_threshold:
                needs_review = False
                return {
                    'label': title_label,
                    'score': title_conf,
                    'needs_review': needs_review,
                    'explanation': (
                        f"FinBERT classified as {title_label.upper()} "
                        f"(confidence {title_conf:.1%}) from the article headline. "
                        f"Distribution — positive {title_agg['positive']:.0%}, "
                        f"neutral {title_agg['neutral']:.0%}, "
                        f"negative {title_agg['negative']:.0%}."
                    ),
                    'focus_mode': 'headline',
                    'cb_mentioned': False,
                    'cb_relevance': 0,
                    'distribution': title_agg,
                    'key_phrases': []
                }

        # ---- Stage 2: CB-focused text ----
        focus_text, focus_mode, cb_mentioned, cb_relevance = self._build_focus_text(text)

        if not focus_text or len(focus_text.strip()) < 50:
            return {
                'label': 'neutral',
                'score': 0.5,
                'needs_review': True,
                'explanation': 'Insufficient content to analyze sentiment.',
                'focus_mode': 'full',
                'cb_mentioned': cb_mentioned,
                'cb_relevance': cb_relevance,
                'distribution': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'key_phrases': []
            }

        chunks = self._chunk_text(focus_text)
        agg, n = self._aggregate_finbert(chunks)

        sentiment = max(agg, key=agg.get)
        confidence = agg[sentiment]
        needs_review = confidence < self.confidence_threshold

        focus_label = {
            'cb_context': "CrossBoundary-specific sentences",
            'pillar': "topic-relevant sentences",
            'full': "the full article body"
        }.get(focus_mode, focus_mode)

        review_note = (
            f" ⚠️ LOW CONFIDENCE (below {self.confidence_threshold:.0%} threshold) — "
            f"recommend manual review before logging."
            if needs_review else ""
        )

        explanation = (
            f"FinBERT classified this as {sentiment.upper()} "
            f"(confidence {confidence:.1%}), scored on {focus_label} "
            f"across {n} decisive chunk(s). "
            f"Class distribution — positive {agg['positive']:.0%}, "
            f"neutral {agg['neutral']:.0%}, negative {agg['negative']:.0%}."
            f"{review_note}"
        )

        return {
            'label': sentiment,
            'score': confidence,
            'needs_review': needs_review,
            'explanation': explanation,
            'focus_mode': focus_mode,
            'cb_mentioned': cb_mentioned,
            'cb_relevance': cb_relevance,
            'distribution': agg,
            'key_phrases': []
        }

    # ------------------------------------------------------------------ #
    # Positioning alignment (now with a 0-100 score + CB boost)
    # ------------------------------------------------------------------ #
    def check_alignment_with_explanation(self, text, cb_mentioned=False):
        """Check positioning alignment with a numeric CB alignment score."""
        if not text:
            return {
                'status': 'NO',
                'score': 0,
                'pillars': [],
                'explanation': 'No content available to analyze positioning.',
                'matches': []
            }

        text_lower = text.lower()
        matches = []
        explanation_parts = []
        raw_score = 0.0

        for pillar, config in self.positioning_pillars.items():
            matched_keywords = self._match_keywords(text_lower, config['keywords'])
            if matched_keywords:
                # diminishing returns: first few hits matter most
                pillar_contrib = config['weight'] * min(len(matched_keywords), 3)
                raw_score += pillar_contrib
                matches.append({
                    'pillar': pillar,
                    'keywords': matched_keywords,
                    'description': config['description']
                })
                explanation_parts.append(
                    f"• {pillar.replace('_', ' ').title()}: Found keywords "
                    f"'{', '.join(matched_keywords[:2])}'"
                )

        # Direct CB mention is a strong positioning signal
        if cb_mentioned:
            raw_score += 2.0

        # Normalize to 0-100 (3 pillars * 3 hits * ~1.0 weight + CB boost ~= 11 cap)
        score = int(min(100, raw_score / 11.0 * 100))

        n_pillars = len(matches)
        if n_pillars >= 2 or (n_pillars >= 1 and cb_mentioned):
            status = 'YES'
            explanation = (f"✅ STRONG ALIGNMENT (score {score}/100). Matched {n_pillars} "
                           f"pillar(s){' + direct CrossBoundary mention' if cb_mentioned else ''}:\n"
                           + "\n".join(explanation_parts))
        elif n_pillars == 1:
            status = 'PARTIAL'
            explanation = (f"🟡 PARTIAL ALIGNMENT (score {score}/100). Matched 1 pillar:\n"
                           + "\n".join(explanation_parts))
        else:
            status = 'NO'
            explanation = ("❌ NO ALIGNMENT detected. Content doesn't match frontier "
                           "investment, climate finance, or energy access language.")

        return {
            'status': status,
            'score': score,
            'pillars': [m['pillar'] for m in matches],
            'explanation': explanation,
            'matches': matches
        }

    # ------------------------------------------------------------------ #
    # Business outcomes (CB mention gate retained)
    # ------------------------------------------------------------------ #
    def determine_outcomes_with_explanation(self, text, sentiment, matched_pillars):
        """Determine business outcomes; only tag when CrossBoundary is mentioned."""
        text_lower = text.lower()
        cb_mentioned = self._cb_sentence_match(text_lower)

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

        if page_data.get('is_paywall') and not page_data['text']:
            return {
                'url': url,
                'title': page_data['title'],
                'status': 'Paywall',
                'error': True,
                'paywall': True,
                'corrupted': False,
                'sentiment': {'label': 'neutral', 'score': 0.0, 'needs_review': True,
                              'cb_mentioned': False, 'cb_relevance': 0, 'focus_mode': 'full',
                              'explanation': 'Paywalled article — manual entry required'},
                'alignment': {'status': 'NO', 'score': 0,
                              'explanation': 'Paywalled — content unavailable', 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False,
                             'explanation': 'Paywalled — manual entry required'},
                'key_sentences': []
            }

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
                'sentiment': {'label': 'neutral', 'score': 0.0, 'needs_review': True,
                              'cb_mentioned': False, 'cb_relevance': 0, 'focus_mode': 'full',
                              'explanation': explanation},
                'alignment': {'status': 'NO', 'score': 0, 'explanation': explanation, 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False, 'explanation': explanation},
                'key_sentences': []
            }

        sentiment = self.analyze_sentiment_with_explanation(page_data['text'], title=page_data['title'])
        alignment = self.check_alignment_with_explanation(
            page_data['text'], cb_mentioned=sentiment.get('cb_mentioned', False)
        )
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
    """Create HTML report with all status indicators surfaced."""
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
        <p><em>v3 — headline-first scoring, drop-ambiguous-chunks aggregation, CB-focused text,
        expanded CrossBoundary taxonomy, CB mention gate, corrupted-content voiding,
        and paywall detection.</em></p>

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
        cb_relevance = result['sentiment'].get('cb_relevance', 0)

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
                <p><strong>Status:</strong> <span class="{alignment_class}">{result['alignment']['status']}</span>
                (Alignment score: {result['alignment'].get('score', 0)}/100 · CB relevance: {cb_relevance}/100)</p>
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
            <p>Report generated by Company Image Sentiment Analyzer v3</p>
            <p>Headline-first scoring · drop-ambiguous-chunks · CB-focused text · expanded CB taxonomy · CB mention gate · corrupted-content voiding · paywall detection</p>
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
        <p>FinBERT-powered, CrossBoundary-tuned analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedCompanyAnalyzer()

    with st.sidebar:
        st.markdown("### ⚙️ Tuning Controls")

        confidence_threshold = st.slider(
            "Confidence threshold for 'Needs Review'",
            min_value=0.40, max_value=0.90, value=0.55, step=0.01,
            help="Results below this confidence are flagged for manual review. "
                 "0.55 is a realistic default for FinBERT on full news articles — "
                 "it typically tops out at 60-65% on mixed-content pages."
        )
        focus_cb = st.toggle(
            "Focus sentiment on CrossBoundary context",
            value=True,
            help="When CB is mentioned, score sentiment on the sentences about CB (plus "
                 "neighbors) rather than the whole page. Makes the score CB-specific."
        )

        max_workers = st.slider("Parallel workers", 1, 3, 1)
        st.caption("Keep at 1 on Streamlit Community Cloud (1 GB RAM) to avoid out-of-memory crashes.")

        st.markdown("---")
        st.markdown("### 📊 How the tuning works")
        st.info(
            "**Headline-first:** if the article title alone crosses the threshold, "
            "that's used as the final score — headlines are FinBERT's strongest input.\n\n"
            "**Drop ambiguous chunks:** chunks where no class reaches 50% are excluded "
            "from averaging — they add noise, not signal. Neutral can still win.\n\n"
            "**CB focus:** sentiment reflects how the article feels about CrossBoundary.\n\n"
            "**CB relevance score:** 0–100 estimate of how much the article is about CB."
        )

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info(
            "- Expand any result for the full explanation\n"
            "- Download CSV and filter 'Needs Review = TRUE' first\n"
            "- Lower the threshold if too many true-positive CB articles get flagged"
        )

    # Push current settings onto the analyzer before running
    analyzer = st.session_state.analyzer
    analyzer.confidence_threshold = confidence_threshold
    analyzer.focus_cb = focus_cb

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
            futures = {executor.submit(analyzer.analyze_url, url): url for url in urls}
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
                'Needs Review': r['sentiment'].get('needs_review', True) if not r.get('error') else True,
                'Focus Mode': r['sentiment'].get('focus_mode', 'full') if not r.get('error') else 'N/A',
                'CB Relevance': r['sentiment'].get('cb_relevance', 0) if not r.get('error') else 0,
                'Alignment': r['alignment']['status'] if not r.get('error') else 'N/A',
                'Alignment Score': r['alignment'].get('score', 0) if not r.get('error') else 0,
                'CB Mentioned': r['outcomes'].get('cb_mentioned', False) if not r.get('error') else False,
                'Outcomes': ', '.join([o['outcome'] for o in r['outcomes']['outcomes']]) if not r.get('error') and r['outcomes']['outcomes'] else '',
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
    """Display results with expandable explanations and status indicators."""
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
        review_count = sum(1 for r in successful if r['sentiment'].get('needs_review', False))
        st.metric("⚠️ Needs Review", review_count)
    with col5:
        avg_confidence = sum(r['sentiment']['score'] for r in successful) / len(successful) if successful else 0
        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")

    st.markdown("---")

    review_articles = [r for r in successful if r['sentiment'].get('needs_review', False)]
    if review_articles:
        with st.expander(f"⚠️ Manual Review Queue ({len(review_articles)} articles need checking)", expanded=True):
            st.markdown("These articles are below the confidence threshold and should be reviewed before logging.")
            for r in review_articles:
                st.markdown(
                    f"- **{r['title'][:70]}** — {r['sentiment']['label'].upper()} "
                    f"({r['sentiment']['score']:.1%} confidence) — [link]({r['url']})"
                )

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

        status_icon = "✅" if not result.get('error') else ("🔒" if paywall_flag else "❌")
        review_tag = " ⚠️" if needs_review_flag else ""
        cb_tag = " 🏢" if cb_flag else ""
        title_display = result['title'][:70] if len(result['title']) > 70 else result['title']

        with st.expander(f"{status_icon} {idx}. {title_display}{review_tag}{cb_tag}", expanded=False):

            if paywall_flag:
                st.error("🔒 Paywalled article — manual entry required")
                st.code(result['url'])
                continue

            if result.get('error'):
                st.error(f"❌ {result['status']}: {result['title']}")
                st.code(result['url'])
                if corrupted_flag:
                    st.warning("💥 Corrupted binary content was detected and voided. Re-run or log manually.")
                continue

            st.markdown(f"**URL:** {result['url']}")

            if cb_flag:
                st.success(
                    f"🏢 CrossBoundary mentioned — outcomes tagged "
                    f"(CB relevance {result['sentiment'].get('cb_relevance', 0)}/100, "
                    f"sentiment scored on: {result['sentiment'].get('focus_mode', 'full')})"
                )
            else:
                st.info("ℹ️ CrossBoundary not mentioned — sector/industry coverage, outcomes not tagged")

            st.markdown("#### 🎭 Sentiment Analysis")
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'orange'
            }.get(result['sentiment']['label'], 'gray')

            box_class = "review-box" if needs_review_flag else "explanation-box"
            review_warning = "<br><strong>⚠️ Low confidence — manually verify before logging.</strong>" if needs_review_flag else ""

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
                {result['alignment']['status']}</span>
                &nbsp;|&nbsp; <strong>Alignment score:</strong> {result['alignment'].get('score', 0)}/100<br>
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
