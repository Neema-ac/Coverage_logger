# Company Image Sentiment Analyzer  (CB-tuned, v4)
# ------------------------------------------------
# Install:
#   pip install streamlit trafilatura transformers torch beautifulsoup4 requests pandas reportlab
#
# Run:
#   streamlit run company_sentiment_analyzer.py
#
# What changed in v4 (three coordinated fixes):
#
#   1. TITLE EXTRACTION FALLBACK (decoupled from body-extraction failure)
#      - Previously the BeautifulSoup/meta title fallback only ran when body
#        extraction nearly failed (len(text) < 50). So when trafilatura got the
#        body but missed the title, articles fell through as "Untitled Article".
#      - Now: whenever trafilatura returns no title, we ALWAYS try the soup
#        fallback (<title> -> og:title -> <h1>), independent of body length.
#        The soup is built once and reused.
#      - Site-name suffixes (" | African Mining Market", " - Semafor") are
#        stripped so the headline is a clean FinBERT input.
#
#   2. HEADLINE-FIRST IS NOW CB-GATED  (required alongside fix #1)
#      - With titles now populated, the old headline-first stage would start
#        firing on whole-article headlines that have nothing to do with CB —
#        reintroducing exactly the "wrong CB sentiment" problem from a different
#        direction. For a CB-SPECIFIC tool that's wrong.
#      - Now: headline-first only triggers when the headline itself contains a
#        CrossBoundary name. Otherwise we skip straight to CB-focused body
#        scoring. Generic topic headlines no longer override CB sentiment.
#
#   3. EXCERPT DISPLAY MATCHES THE SCORED TEXT
#      - Previously "Key Excerpts" showed key_sentences (first long sentences of
#        the body) while FinBERT scored a SEPARATE focus_text. Reviewers saw text
#        that wasn't scored, making correct labels look wrong.
#      - Now the sentiment result carries the actual scored span ('scored_text'
#        + 'scored_excerpts'), and the report surfaces THAT under a clearly
#        labelled "Scored text (what FinBERT actually read)" section. The raw
#        body excerpts are still shown separately for context.
#
# Carried over from v3: CB-focused sentiment, 0-100 CB alignment + relevance,
# drop-ambiguous-chunks aggregation, expanded CB taxonomy, CB mention gate on
# outcomes, corrupted-content voiding, paywall detection, confidence flagging.

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
.scored-box {
    background-color: #eef4ff;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #3b6fe0;
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

# Common site-name suffix separators to strip from <title> tags
_TITLE_SUFFIX_RE = re.compile(r'\s*[|\u2013\u2014\-]\s*[^|\u2013\u2014\-]{2,40}$')

# ------------------------------------------------------------------ #
# Coverage type classification constants
# ------------------------------------------------------------------ #

# CrossBoundary-owned publishing domains
CB_OWNED_DOMAINS = [
    "crossboundary.com", "crossboundaryenergy.com",
    "crossboundary.energy", "crossboundaryaccess.com",
]

# Paid PR wire / press release distribution services → treated as Owned
PR_WIRE_DOMAINS = [
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    "accesswire.com", "prweb.com", "einpresswire.com", "newswire.com",
    "presswire.com", "send2press.com",
]

# Proactive signal patterns — CB-initiated coverage indicators
# Checked against lowercase article body text
PROACTIVE_PATTERNS = [
    # Formal announcement language
    (r'\b(?:announces|announced|today announced|is pleased to announce)\b',
     "announcement language"),
    # "About CrossBoundary" company boilerplate — classic press release footer
    (r'\babout crossboundary\b',
     "company boilerplate ('About CrossBoundary') detected"),
    # Press release headers / footers
    (r'\bfor immediate release\b',
     "press release header detected"),
    (r'\bpress release\b',
     "press release label detected"),
    # CB executive directly quoted: "said [Name], [title]... CrossBoundary"
    # or "CrossBoundary's [title] said"
    (r'said\s+\w[\w\s]{2,30},\s*(?:ceo|cfo|coo|founder|co-founder|director|'
     r'head|partner|manager|associate|analyst)\s+(?:of\s+|at\s+)?crossboundary',
     "CB executive quoted"),
    (r'crossboundary(?:\s+energy|\s+advisory|\s+access|\s+group)?'
     r'[\'s]*\s+(?:ceo|cfo|coo|founder|co-founder|director|head|partner)',
     "CB executive title attributed"),
    # Generic "crossboundary said" / "crossboundary noted"
    (r'crossboundary(?:\s+energy|\s+advisory|\s+access|\s+group)?\s+'
     r'(?:said|noted|stated|confirmed|commented|added)',
     "CB organisation statement"),
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
    @staticmethod
    def _clean_title(title):
        """Normalise whitespace, strip a trailing ' | Site Name' suffix, cap length."""
        if not title:
            return None
        title = re.sub(r'\s+', ' ', title).strip()
        # Strip a single trailing site-name suffix if the remaining title is substantial
        stripped = _TITLE_SUFFIX_RE.sub('', title).strip()
        if len(stripped) >= 15:
            title = stripped
        return title[:120] or None

    def extract_title(self, soup):
        """Extract article title from a BeautifulSoup tree (fallback path).
        Order: <title> -> og:title -> twitter:title -> <h1>."""
        candidates = []

        if soup.title and soup.title.string:
            candidates.append(soup.title.string)

        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            candidates.append(og_title.get('content'))

        tw_title = soup.find('meta', attrs={'name': 'twitter:title'})
        if tw_title and tw_title.get('content'):
            candidates.append(tw_title.get('content'))

        h1 = soup.find('h1')
        if h1 and h1.get_text(strip=True):
            candidates.append(h1.get_text())

        for cand in candidates:
            cleaned = self._clean_title(cand)
            if cleaned:
                return cleaned
        return None

    def fetch_page_content(self, url):
        """Fetch a page and extract the clean ARTICLE BODY using trafilatura,
        falling back to a cleaned BeautifulSoup parse if needed.
        Includes corrupted-content and paywall detection.

        TITLE FALLBACK (v4): whenever trafilatura does not supply a title, the
        BeautifulSoup/meta fallback runs regardless of whether body extraction
        succeeded. The soup tree is built at most once and reused."""
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

            title = None
            text = ""
            corrupted = False
            soup = None  # built lazily, reused across title + body fallback

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
                title = self._clean_title(data.get('title'))

            # ---- TITLE FALLBACK (decoupled from body success) ----
            # Run whenever trafilatura gave us no usable title, independent of
            # whether the body extracted fine.
            if not title and html:
                soup = BeautifulSoup(html, 'html.parser')
                title = self.extract_title(soup)

            # ---- BODY FALLBACK (only when trafilatura body nearly failed) ----
            if len(text) < 50:
                if soup is None:
                    soup = BeautifulSoup(html, 'html.parser')
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

            title = title or "Untitled Article"

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

    def _headline_mentions_cb(self, headline):
        """True if the headline text itself names CrossBoundary (gates headline-first)."""
        if not headline:
            return False
        return self._cb_sentence_match(headline.lower())

    def _is_cb_mentioned(self, text):
        """Check if CrossBoundary is mentioned anywhere in the article body.
        A single mention — including a quoted CB member statement — counts."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return any(self._cb_sentence_match(s.lower()) for s in sentences)

    # ------------------------------------------------------------------ #
    # Coverage type classification
    # ------------------------------------------------------------------ #
    def classify_coverage_type(self, url, text, cb_mentioned):
        """Classify coverage as Owned / Proactive / Earned / Unknown."""
        url_lower = (url or "").lower()
        text_lower = (text or "").lower()

        if any(domain in url_lower for domain in CB_OWNED_DOMAINS):
            return {'type': 'Owned', 'icon': '🏠',
                    'reason': 'Published on a CrossBoundary-owned domain'}

        if any(domain in url_lower for domain in PR_WIRE_DOMAINS):
            return {'type': 'Owned', 'icon': '🏠',
                    'reason': 'Distributed via a paid PR wire service'}

        if not text_lower:
            return {'type': 'Unknown', 'icon': '❓',
                    'reason': 'No article content available (paywalled or failed extraction)'}

        if not cb_mentioned:
            return {'type': 'Unknown', 'icon': '❓',
                    'reason': 'CrossBoundary not mentioned — coverage type indeterminate'}

        for pattern, label in PROACTIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return {'type': 'Proactive', 'icon': '📣',
                        'reason': f'CB-initiated signal: {label}'}

        return {'type': 'Earned', 'icon': '📰',
                'reason': 'CB mentioned in independent third-party coverage'}

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

        cb_mentioned = len(cb_idx) >= 1

        if cb_mentioned:
            density = (len(cb_idx) + 0.5 * len(set(pillar_idx) - set(cb_idx))) / max(len(sentences), 1)
            cb_relevance = int(min(100, 50 + density * 200))
        elif pillar_idx:
            cb_relevance = int(min(45, len(pillar_idx) / max(len(sentences), 1) * 180))
        else:
            cb_relevance = 0

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

        if len(focus_text.strip()) < 300:
            focus_text = text
            mode = 'full'

        return focus_text, mode, cb_mentioned, cb_relevance

    # ------------------------------------------------------------------ #
    # FinBERT aggregation (drop-ambiguous-chunks; neutral fully preserved)
    # ------------------------------------------------------------------ #
    def _aggregate_finbert(self, chunks):
        """Run FinBERT on each chunk and combine into one distribution.

        Returns (agg_distribution, n_decisive, decisive_chunks) where
        decisive_chunks is the list of chunk strings that actually fed the score
        (used to show the user exactly what was scored)."""
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

        decisive = [(ch, d) for ch, d, conf in per_chunk_data if conf >= 0.50]
        if not decisive:
            best = max(per_chunk_data, key=lambda x: x[2])
            decisive = [(best[0], best[1])]

        agg = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        total_w = 0.0
        for ch, dist in decisive:
            w = float(len(ch)) + 1e-6
            total_w += w
            for k in agg:
                agg[k] += dist[k] * w
        for k in agg:
            agg[k] /= total_w

        decisive_chunks = [ch for ch, _ in decisive]
        return agg, len(decisive), decisive_chunks

    @staticmethod
    def _excerpts_from(scored_text, max_excerpts=3):
        """Pull a few representative sentences from the actually-scored text,
        for display so reviewers see what FinBERT read."""
        if not scored_text:
            return []
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', scored_text) if s.strip()]
        # Prefer substantive sentences; fall back to whatever exists
        substantive = [s for s in sents if len(s) > 40] or sents
        return substantive[:max_excerpts]

    # ------------------------------------------------------------------ #
    # Sentiment (CB-focused + threshold from settings)
    # ------------------------------------------------------------------ #
    _SKIP_TITLES = {"untitled article", "untitled", ""}

    def analyze_sentiment_with_explanation(self, text, title=""):
        """3-class sentiment via FinBERT.

        Stage 1: headline-first — NOW CB-GATED. Only used when the headline is a
        real title, is confident (>= threshold), AND the headline itself names
        CrossBoundary. This keeps a CB-specific tool from logging a generic
        topic-headline sentiment as if it were CB sentiment.

        Stage 2: CB-focused text (or full-article fallback).

        The returned dict always includes 'scored_text' and 'scored_excerpts'
        describing exactly what FinBERT read, so the report can display it.
        """
        focus_text, focus_mode, cb_mentioned, cb_relevance = self._build_focus_text(text)

        # ---- Stage 1: headline-first (CB-gated) ----
        headline = (title or "").strip()
        headline_is_real = (
            len(headline) > 15
            and headline.lower() not in self._SKIP_TITLES
            and not headline.lower().startswith("error:")
        )
        if headline_is_real and self._headline_mentions_cb(headline):
            title_agg, _, _ = self._aggregate_finbert([headline])
            title_label = max(title_agg, key=title_agg.get)
            title_conf = title_agg[title_label]
            if title_conf >= self.confidence_threshold:
                return {
                    'label': title_label,
                    'score': title_conf,
                    'needs_review': False,
                    'explanation': (
                        f"FinBERT classified as {title_label.upper()} "
                        f"(confidence {title_conf:.1%}) from the article headline, "
                        f"which directly names CrossBoundary. "
                        f"Distribution — positive {title_agg['positive']:.0%}, "
                        f"neutral {title_agg['neutral']:.0%}, "
                        f"negative {title_agg['negative']:.0%}."
                    ),
                    'focus_mode': 'headline',
                    'cb_mentioned': cb_mentioned,
                    'cb_relevance': cb_relevance,
                    'distribution': title_agg,
                    'scored_text': headline,
                    'scored_excerpts': [headline],
                    'key_phrases': []
                }

        # ---- Stage 2: CB-focused text ----
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
                'scored_text': '',
                'scored_excerpts': [],
                'key_phrases': []
            }

        chunks = self._chunk_text(focus_text)
        agg, n, decisive_chunks = self._aggregate_finbert(chunks)

        sentiment = max(agg, key=agg.get)
        confidence = agg[sentiment]
        needs_review = confidence < self.confidence_threshold

        scored_text = " ".join(decisive_chunks)
        scored_excerpts = self._excerpts_from(scored_text)

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
            'scored_text': scored_text,
            'scored_excerpts': scored_excerpts,
            'key_phrases': []
        }

    # ------------------------------------------------------------------ #
    # Positioning alignment (0-100 score + CB boost)
    # ------------------------------------------------------------------ #
    def check_alignment_with_explanation(self, text, cb_mentioned=False):
        if not text:
            return {'status': 'NO', 'score': 0, 'pillars': [],
                    'explanation': 'No content available to analyze positioning.', 'matches': []}

        text_lower = text.lower()
        matches = []
        explanation_parts = []
        raw_score = 0.0

        for pillar, config in self.positioning_pillars.items():
            matched_keywords = self._match_keywords(text_lower, config['keywords'])
            if matched_keywords:
                pillar_contrib = config['weight'] * min(len(matched_keywords), 3)
                raw_score += pillar_contrib
                matches.append({'pillar': pillar, 'keywords': matched_keywords,
                                'description': config['description']})
                explanation_parts.append(
                    f"• {pillar.replace('_', ' ').title()}: Found keywords "
                    f"'{', '.join(matched_keywords[:2])}'"
                )

        if cb_mentioned:
            raw_score += 2.0

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

        return {'status': status, 'score': score,
                'pillars': [m['pillar'] for m in matches],
                'explanation': explanation, 'matches': matches}

    # ------------------------------------------------------------------ #
    # Business outcomes (CB mention gate retained)
    # ------------------------------------------------------------------ #
    def determine_outcomes_with_explanation(self, text, sentiment, matched_pillars):
        text_lower = text.lower()
        cb_mentioned = self._is_cb_mentioned(text)

        if not cb_mentioned:
            return {'outcomes': [], 'cb_mentioned': False,
                    'explanation': ("CrossBoundary not mentioned in this article — "
                                    "business outcomes not tagged for sector/industry coverage.")}

        outcomes = []
        for outcome, config in self.business_outcomes.items():
            matched_keywords = self._match_keywords(text_lower, config['keywords'])
            if matched_keywords:
                if outcome == "Award/Recognition" and sentiment != 'positive':
                    continue
                outcomes.append({
                    'outcome': outcome, 'keywords': matched_keywords,
                    'description': config['description'],
                    'confidence': min(len(matched_keywords) / len(config['keywords']) * 100, 100)
                })

        outcomes.sort(key=lambda x: x['confidence'], reverse=True)
        top_outcomes = outcomes[:3]

        if not top_outcomes and matched_pillars and sentiment == 'positive':
            top_outcomes = [{'outcome': 'Investor Narrative', 'keywords': [],
                             'description': 'Strengthens investor confidence or fundraising narrative',
                             'confidence': 60}]

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

        return {'outcomes': top_outcomes, 'cb_mentioned': True, 'explanation': explanation}

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def analyze_url(self, url):
        page_data = self.fetch_page_content(url)

        if page_data.get('is_paywall') and not page_data['text']:
            return {
                'url': url, 'title': page_data['title'], 'status': 'Paywall',
                'error': True, 'paywall': True, 'corrupted': False,
                'sentiment': {'label': 'neutral', 'score': 0.0, 'needs_review': True,
                              'cb_mentioned': False, 'cb_relevance': 0, 'focus_mode': 'full',
                              'scored_text': '', 'scored_excerpts': [],
                              'explanation': 'Paywalled article — manual entry required'},
                'alignment': {'status': 'NO', 'score': 0,
                              'explanation': 'Paywalled — content unavailable', 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False,
                             'explanation': 'Paywalled — manual entry required'},
                'coverage': {'type': 'Unknown', 'icon': '❓',
                             'reason': 'Paywalled — content unavailable'},
                'key_sentences': []
            }

        if page_data.get('corrupted') or not page_data['text']:
            status = 'Corrupted' if page_data.get('corrupted') else 'Failed'
            explanation = (
                'Corrupted binary content detected — result voided. Re-run or log manually.'
                if page_data.get('corrupted') else 'Could not fetch content.'
            )
            return {
                'url': url, 'title': page_data['title'], 'status': status,
                'error': True, 'paywall': False, 'corrupted': page_data.get('corrupted', False),
                'sentiment': {'label': 'neutral', 'score': 0.0, 'needs_review': True,
                              'cb_mentioned': False, 'cb_relevance': 0, 'focus_mode': 'full',
                              'scored_text': '', 'scored_excerpts': [],
                              'explanation': explanation},
                'alignment': {'status': 'NO', 'score': 0, 'explanation': explanation, 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False, 'explanation': explanation},
                'coverage': {'type': 'Unknown', 'icon': '❓', 'reason': explanation},
                'key_sentences': []
            }

        sentiment = self.analyze_sentiment_with_explanation(page_data['text'], title=page_data['title'])
        alignment = self.check_alignment_with_explanation(
            page_data['text'], cb_mentioned=sentiment.get('cb_mentioned', False)
        )
        outcomes = self.determine_outcomes_with_explanation(
            page_data['text'], sentiment['label'], alignment['pillars']
        )
        coverage = self.classify_coverage_type(
            url, page_data['text'], cb_mentioned=sentiment.get('cb_mentioned', False)
        )

        return {
            'url': url, 'title': page_data['title'], 'status': 'Success',
            'error': False, 'paywall': False, 'corrupted': False,
            'sentiment': sentiment, 'alignment': alignment, 'outcomes': outcomes,
            'coverage': coverage, 'key_sentences': page_data['key_sentences'][:5],
            'text_preview': page_data['text'][:300] + "..."
        }


# ------------------------------------------------------------------ #
# HTML Report
# ------------------------------------------------------------------ #
def create_html_report(results):
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
            .scored {{ background: #eef4ff; padding: 15px; border-left: 4px solid #3b6fe0; margin: 10px 0; }}
            .needs-review {{ background: #fff8e1; padding: 15px; border-left: 4px solid #f39c12; margin: 10px 0; }}
            .error-paywall {{ background: #fff3f3; padding: 15px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
            .keyword {{ background: #fff3cd; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            .badge-review {{ background: #f39c12; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; }}
            .badge-paywall {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; }}
            .badge-cb {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px; }}
            .coverage-owned {{ color: #2980b9; font-weight: bold; }}
            .coverage-proactive {{ color: #8e44ad; font-weight: bold; }}
            .coverage-earned {{ color: #27ae60; font-weight: bold; }}
            .coverage-unknown {{ color: #95a5a6; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🏢 Company Image Sentiment Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><em>v4 — title fallback, CB-gated headline-first, scored-text excerpt display,
        CB-focused text, expanded CrossBoundary taxonomy, CB mention gate,
        corrupted-content voiding, paywall detection.</em></p>

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
        owned_count = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Owned')
        proactive_count = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Proactive')
        earned_count = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Earned')

        html_content += f"<p><strong>Positive Sentiment:</strong> {pos_count}</p>"
        html_content += f"<p><strong>Negative Sentiment:</strong> {neg_count}</p>"
        html_content += f"<p><strong>Neutral Sentiment:</strong> {neu_count}</p>"
        html_content += f"<p><strong>Strong Alignment:</strong> {aligned_count}</p>"
        html_content += f"<p><strong>🏠 Owned:</strong> {owned_count} &nbsp; <strong>📣 Proactive:</strong> {proactive_count} &nbsp; <strong>📰 Earned:</strong> {earned_count}</p>"

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
        focus_mode = result['sentiment'].get('focus_mode', 'full')
        coverage = result.get('coverage', {'type': 'Unknown', 'icon': '❓', 'reason': ''})
        coverage_class = f"coverage-{coverage['type'].lower()}"

        review_badge = '<span class="badge-review">⚠️ NEEDS REVIEW</span>' if needs_review_flag else ''
        cb_badge = '<span class="badge-cb">CB MENTIONED</span>' if cb_flag else ''

        focus_label = {
            'cb_context': "CrossBoundary-specific sentences",
            'pillar': "topic-relevant sentences",
            'full': "the full article body",
            'headline': "the article headline"
        }.get(focus_mode, focus_mode)

        html_content += f"""
        <div class="article">
            <h3>{idx}. {result['title']} {review_badge} {cb_badge}</h3>
            <p><strong>URL:</strong> <a href="{result['url']}">{result['url']}</a></p>

            <h4>📡 Coverage Type</h4>
            <div class="explanation">
                <p><strong>Type:</strong> <span class="{coverage_class}">{coverage['icon']} {coverage['type'].upper()}</span></p>
                <p><strong>Reason:</strong> {coverage['reason']}</p>
            </div>

            <h4>🎭 Sentiment Analysis</h4>
            <div class="{'needs-review' if needs_review_flag else 'explanation'}">
                <p><strong>Result:</strong> <span class="{sentiment_class}">{result['sentiment']['label'].upper()}</span>
                (Confidence: {result['sentiment']['score']:.1%})
                {'<strong> ⚠️ Low confidence — manual review recommended before logging.</strong>' if needs_review_flag else ''}</p>
                <p><strong>Explanation:</strong> {result['sentiment']['explanation']}</p>
            </div>
        """

        # ---- Scored text: exactly what FinBERT read (v4) ----
        scored_excerpts = result['sentiment'].get('scored_excerpts', [])
        if scored_excerpts:
            html_content += f"""
            <h4>🔬 Scored Text <span style="font-weight:normal;font-size:0.85em;color:#555;">
            (what FinBERT actually read — {focus_label})</span></h4>
            <div class="scored">
                <ul>
            """
            for s in scored_excerpts:
                html_content += f"<li>\"{s[:240]}\"</li>"
            html_content += "</ul></div>"

        html_content += f"""
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
            <h4>📝 Article Body Excerpts <span style="font-weight:normal;font-size:0.85em;color:#555;">
            (context only — not necessarily the scored text)</span></h4>
            <ul>
            """
            for sentence in result['key_sentences'][:3]:
                html_content += f"<li>\"{sentence[:200]}...\"</li>"
            html_content += "</ul>"

        html_content += "</div>"

    html_content += """
        <div class="footer">
            <p>Report generated by Company Image Sentiment Analyzer v4</p>
            <p>Title fallback · CB-gated headline-first · scored-text display · CB-focused text · expanded CB taxonomy · CB mention gate · corrupted-content voiding · paywall detection</p>
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
            "**Headline-first (CB-gated):** if the headline NAMES CrossBoundary and "
            "FinBERT is confident on it, that's the score. Generic topic headlines "
            "no longer override CB sentiment.\n\n"
            "**Drop ambiguous chunks:** chunks where no class reaches 50% are excluded "
            "from averaging — they add noise, not signal. Neutral can still win.\n\n"
            "**CB focus:** sentiment reflects how the article feels about CrossBoundary.\n\n"
            "**Scored text shown:** the report displays the exact text FinBERT read.\n\n"
            "**CB relevance score:** 0–100 estimate of how much the article is about CB."
        )

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info(
            "- Expand any result for the full explanation\n"
            "- Check '🔬 Scored Text' to see exactly what FinBERT read\n"
            "- Download CSV and filter 'Needs Review = TRUE' first\n"
            "- Lower the threshold if too many true-positive CB articles get flagged"
        )

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
                'Scored Text': r['sentiment'].get('scored_text', '')[:500] if not r.get('error') else '',
                'CB Relevance': r['sentiment'].get('cb_relevance', 0) if not r.get('error') else 0,
                'Alignment': r['alignment']['status'] if not r.get('error') else 'N/A',
                'Alignment Score': r['alignment'].get('score', 0) if not r.get('error') else 0,
                'CB Mentioned': r['outcomes'].get('cb_mentioned', False) if not r.get('error') else False,
                'Coverage Type': r.get('coverage', {}).get('type', 'Unknown'),
                'Coverage Reason': r.get('coverage', {}).get('reason', ''),
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

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        owned = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Owned')
        st.metric("🏠 Owned", owned)
    with col_b:
        proactive = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Proactive')
        st.metric("📣 Proactive", proactive)
    with col_c:
        earned = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Earned')
        st.metric("📰 Earned", earned)
    with col_d:
        unknown_cov = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Unknown')
        st.metric("❓ Unknown", unknown_cov)

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

            coverage = result.get('coverage', {'type': 'Unknown', 'icon': '❓', 'reason': ''})
            cov_colors = {'Owned': '#2980b9', 'Proactive': '#8e44ad',
                          'Earned': '#27ae60', 'Unknown': '#95a5a6'}
            cov_color = cov_colors.get(coverage['type'], '#95a5a6')
            st.markdown(f"""
            <div class="explanation-box">
                <strong>📡 Coverage Type:</strong>
                <span style="color:{cov_color}; font-weight:bold;">
                {coverage['icon']} {coverage['type'].upper()}</span>
                &nbsp;—&nbsp; {coverage['reason']}
            </div>
            """, unsafe_allow_html=True)

            focus_mode = result['sentiment'].get('focus_mode', 'full')
            if cb_flag:
                st.success(
                    f"🏢 CrossBoundary mentioned — outcomes tagged "
                    f"(CB relevance {result['sentiment'].get('cb_relevance', 0)}/100, "
                    f"sentiment scored on: {focus_mode})"
                )
            else:
                st.info("ℹ️ CrossBoundary not mentioned — sector/industry coverage, outcomes not tagged")

            st.markdown("#### 🎭 Sentiment Analysis")
            sentiment_color = {
                'positive': 'green', 'negative': 'red', 'neutral': 'orange'
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

            # ---- Scored text: exactly what FinBERT read (v4) ----
            scored_excerpts = result['sentiment'].get('scored_excerpts', [])
            focus_label = {
                'cb_context': "CrossBoundary-specific sentences",
                'pillar': "topic-relevant sentences",
                'full': "the full article body",
                'headline': "the article headline"
            }.get(focus_mode, focus_mode)
            if scored_excerpts:
                st.markdown(f"#### 🔬 Scored Text — what FinBERT actually read ({focus_label})")
                for i, s in enumerate(scored_excerpts, 1):
                    st.markdown(
                        f"<div class='scored-box'><strong>{i}.</strong> \"{s[:280]}\"</div>",
                        unsafe_allow_html=True
                    )

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
                st.markdown("#### 📝 Article Body Excerpts (context only — not necessarily the scored text)")
                for i, sentence in enumerate(result['key_sentences'][:3], 1):
                    st.markdown(
                        f"<div class='explanation-box'><strong>{i}.</strong> \"{sentence[:250]}...\"</div>",
                        unsafe_allow_html=True
                    )


if __name__ == "__main__":
    main()
