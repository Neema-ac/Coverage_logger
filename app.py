# Company Image Sentiment Analyzer  (CB-tuned, v8 — syndicated-PR detection)
# ------------------------------------------------------------------
# Install:
#   pip install streamlit trafilatura transformers torch beautifulsoup4 requests pandas reportlab fpdf2 anthropic openai
#
# Run:
#   streamlit run app.py
#
# v8 change: coverage-type classification now catches SYNDICATED press
#   releases — a CB release reposted verbatim on a third-party news site is
#   routed to Proactive instead of being mislabeled Earned. Because the two
#   LLM engines (Claude, Gemini) delegate coverage typing to this file's
#   classify_coverage_type(), this single change fixes all three engines.
#
# v7 change: the analysis "engine" is now selectable in the sidebar.
#   - Gemini (free tier)  -> free LLM, matches your Claude artifact closely
#   - Claude (paid)       -> same reputational-sentiment prompt as the artifact
#   - FinBERT (local)     -> the original, fully offline, no API
# The LLM engines live in claude_engine.py and free_engine.py and return the
# SAME result shape as FinBERT, so every export / UI feature is unchanged.
#
# v6 change: in addition to analyzing URLs, you can now paste raw article
# text and run the SAME analysis pipeline on it (sentiment, positioning
# alignment, business outcomes, coverage type). See `analyze_text` and the
# "Pasted article text" input mode in the UI.

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

# --- v7: pluggable LLM engines (safe to import even if SDKs aren't installed;
#         each module guards its own optional dependency) ---
from claude_engine import ClaudeAnalyzer
from free_engine import FreeAnalyzer

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

# PDF export via fpdf2 — pure-Python, NO system dependencies (no Cairo /
# pyHanko build chain), so it installs cleanly on Streamlit Community Cloud.
# Add `fpdf2` to requirements.txt to enable the "Download PDF" button.
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

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
# ------------------------------------------------------------------ #
CB_NAMES = [
    "crossboundary", "cross boundary", "cross-boundary",
    "crossboundary group", "crossboundary advisory", "crossboundary energy",
    "crossboundary energy access", "crossboundary access",
    "cb energy", "cb advisory", "cb access",
    "cbe", "cbea",
    "fund for nature",
]

CB_AMBIGUOUS = {"cbe", "cbea"}
CB_CONTEXT_TERMS = [
    "energy", "mini-grid", "minigrid", "solar", "africa", "renewable",
    "power", "access", "crossboundary", "fund", "grid"
]

# Common site-name suffix separators to strip from <title> tags
_TITLE_SUFFIX_RE = re.compile(r'\s*[|\u2013\u2014\-]\s*[^|\u2013\u2014\-]{2,40}$')

# ------------------------------------------------------------------ #
# FINANCE SENTIMENT GUARD (v5)
# FinBERT was trained on financial news where "debt", "raised debt", etc.
# skew negative even when the event is a capital-raise win. These cue lists
# let us detect that pattern at the chunk level and conservatively nudge a
# false-negative toward neutral. Only fires when capital-raise cues are present
# AND no genuine-negative cue is present. Kept transparent + toggleable.
# ------------------------------------------------------------------ #
FINANCE_POSITIVE_CUES = [
    "raised", "secured", "closed", "oversubscribed", "record",
    "largest", "landmark", "milestone", "anchored", "capital raise",
    "fundraise", "commitment", "committed", "expansion", "scaling",
    "first", "debt and equity", "debt from", "equity from",
]
GENUINE_NEGATIVE_CUES = [
    "loss", "losses", "default", "defaulted", "lawsuit", "fraud",
    "write-down", "writedown", "impairment", "layoff", "layoffs",
    "bankruptcy", "insolvency", "collapse", "plunge", "slump",
    "downgrade", "missed", "shortfall", "scandal", "probe",
    "investigation", "halt", "halted", "cancelled", "canceled",
    "delay", "delayed", "decline", "fell", "drop", "crisis",
]

# ------------------------------------------------------------------ #
# Coverage type classification constants
# ------------------------------------------------------------------ #
CB_OWNED_DOMAINS = [
    "crossboundary.com", "crossboundaryenergy.com",
    "crossboundary.energy", "crossboundaryaccess.com",
]

PR_WIRE_DOMAINS = [
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    "accesswire.com", "prweb.com", "einpresswire.com", "newswire.com",
    "presswire.com", "send2press.com",
    # v8: APO Group distribution domains (the wire CB uses across Africa)
    "apo-opa.com", "africa-newsroom.com",
]

PROACTIVE_PATTERNS = [
    (r'\b(?:announces|announced|today announced|is pleased to announce)\b',
     "announcement language"),
    (r'\babout crossboundary\b',
     "company boilerplate ('About CrossBoundary') detected"),
    (r'\bfor immediate release\b',
     "press release header detected"),
    (r'\bpress release\b',
     "press release label detected"),
    (r'said\s+\w[\w\s]{2,30},\s*(?:ceo|cfo|coo|founder|co-founder|director|'
     r'head|partner|manager|associate|analyst)\s+(?:of\s+|at\s+)?crossboundary',
     "CB executive quoted"),
    (r'crossboundary(?:\s+energy|\s+advisory|\s+access|\s+group)?'
     r'[\'s]*\s+(?:ceo|cfo|coo|founder|co-founder|director|head|partner)',
     "CB executive title attributed"),
    (r'crossboundary(?:\s+energy|\s+advisory|\s+access|\s+group)?\s+'
     r'(?:said|noted|stated|confirmed|commented|added)',
     "CB organisation statement"),
]

# ------------------------------------------------------------------ #
# Syndicated-PR detection (v8)
# A release reposted VERBATIM on a third-party news site is NOT earned
# editorial — it's CB-initiated content living on someone else's domain.
# These BODY-TEXT cues catch reposts the URL-based PR_WIRE_DOMAINS check
# misses (the repost URL is the news site, not the wire). High-precision
# only, so a single hit can route to Proactive without misfiling earned.
# ------------------------------------------------------------------ #
SYNDICATION_WIRE_CUES = [
    (r'\bdistributed by\b[^.]{0,40}\bapo group\b', "wire credit: 'Distributed by APO Group'"),
    (r'\bapo group\b',                              "wire service named (APO Group)"),
    (r'\bsource\s*:\s*apo\b',                       "wire credit: 'Source: APO'"),
    (r'\bdistributed by\b',                         "syndication credit: 'Distributed by …'"),
    (r'\bissued by\b[^.]{0,40}crossboundary',       "release attribution: 'issued by … CrossBoundary'"),
    (r'\bthis press release\b',                      "self-referential 'this press release'"),
]

# Media/press-contact block — editorial outlets strip these; PR reposts keep
# them. Only treated as a syndication signal when a CrossBoundary contact is
# also present (see _detect_syndicated_pr), so it stays high-precision.
SYNDICATION_CONTACT_PATTERNS = [
    (r'media\s+(?:contact|enquiries|inquiries|relations)\b', "media-contact block"),
    (r'press\s+(?:contact|office|enquiries|inquiries)\b',     "press-contact block"),
    (r'for\s+(?:media|press)\s+(?:enquiries|inquiries|queries)\b', "press-enquiries line"),
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
        self.neutral_margin = 0.10         # v5: pos/neg within this of neutral => neutral
        self.finance_guard = True          # v5: nudge debt/raise false-negatives

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

        self._all_pillar_keywords = [
            kw for cfg in self.positioning_pillars.values() for kw in cfg["keywords"]
        ]

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
        if not title:
            return None
        title = re.sub(r'\s+', ' ', title).strip()
        stripped = _TITLE_SUFFIX_RE.sub('', title).strip()
        if len(stripped) >= 15:
            title = stripped
        return title[:120] or None

    def extract_title(self, soup):
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
            soup = None

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

            if not title and html:
                soup = BeautifulSoup(html, 'html.parser')
                title = self.extract_title(soup)

            if len(text) < 50:
                if soup is None:
                    soup = BeautifulSoup(html, 'html.parser')
                for element in soup(['script', 'style', 'nav', 'footer',
                                     'header', 'iframe', 'aside', 'form']):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)

            text = re.sub(r'\s+', ' ', text).strip()

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
    def _truncate_words(text, max_chars):
        """v5: truncate at a word boundary so display excerpts don't cut
        mid-token (e.g. 'equity fro'). Adds an ellipsis if shortened."""
        if not text or len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        sp = cut.rfind(' ')
        if sp > 0:
            cut = cut[:sp]
        return cut.rstrip(' ,;:') + '…'

    @staticmethod
    def _match_keywords(text_lower, keywords):
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
        if not headline:
            return False
        return self._cb_sentence_match(headline.lower())

    def _is_cb_mentioned(self, text):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return any(self._cb_sentence_match(s.lower()) for s in sentences)

    # ------------------------------------------------------------------ #
    # Syndicated-PR detection (v8)
    # ------------------------------------------------------------------ #
    def _detect_syndicated_pr(self, text_lower):
        """Detect a syndicated/republished CB release on a third-party domain.
        Returns a reason string if detected, else None. High-precision cues
        only, so genuine earned coverage isn't misrouted to Proactive."""
        # Strong, single-hit cue: explicit wire / syndication attribution.
        for pattern, label in SYNDICATION_WIRE_CUES:
            if re.search(pattern, text_lower):
                return label
        # Weaker cue: a contact block, but ONLY when it points back to CB
        # (a CrossBoundary email/domain present) — that combination is PR,
        # not editorial.
        has_cb_contact = (
            bool(re.search(r'crossboundary[\w.\-]*@|@[\w.\-]*crossboundary', text_lower))
            or 'crossboundary.com' in text_lower
        )
        if has_cb_contact:
            for pattern, label in SYNDICATION_CONTACT_PATTERNS:
                if re.search(pattern, text_lower):
                    return f"{label} with a CrossBoundary contact"
        return None

    # ------------------------------------------------------------------ #
    # Coverage type classification
    # ------------------------------------------------------------------ #
    def classify_coverage_type(self, url, text, cb_mentioned):
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

        # v8: syndicated-PR catch — a verbatim release reposted on a third-
        # party site. Runs BEFORE the Earned fallback so reposts don't inflate
        # the Earned (independent-journalism) count.
        syndication_reason = self._detect_syndicated_pr(text_lower)
        if syndication_reason:
            return {'type': 'Proactive', 'icon': '📣',
                    'reason': f'Syndicated press release: {syndication_reason}'}

        return {'type': 'Earned', 'icon': '📰',
                'reason': 'CB mentioned in independent third-party coverage'}

    # ------------------------------------------------------------------ #
    # CB-focused sentence selection
    # ------------------------------------------------------------------ #
    def _build_focus_text(self, text):
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
    # v5: finance false-negative guard (chunk-level, transparent)
    # ------------------------------------------------------------------ #
    def _finance_guard_adjust(self, chunk, dist):
        """If FinBERT calls a chunk negative purely on capital-raise language
        (debt/equity raised, secured, record, etc.) with NO genuine-negative
        cue present, shift part of the negative mass to neutral.

        Returns (adjusted_dist, fired_bool). Conservative: moves at most half
        of the negative mass, never invents positive sentiment."""
        if not self.finance_guard:
            return dist, False
        if max(dist, key=dist.get) != 'negative':
            return dist, False

        cl = chunk.lower()
        has_pos_cue = any(re.search(r'\b' + re.escape(c) + r'\b', cl)
                          for c in FINANCE_POSITIVE_CUES)
        has_neg_cue = any(re.search(r'\b' + re.escape(c) + r'\b', cl)
                          for c in GENUINE_NEGATIVE_CUES)

        if not (has_pos_cue and not has_neg_cue):
            return dist, False

        adj = dict(dist)
        shift = adj['negative'] * 0.5
        adj['negative'] -= shift
        adj['neutral'] += shift
        # renormalise defensively
        total = sum(adj.values()) or 1.0
        for k in adj:
            adj[k] /= total
        return adj, True

    # ------------------------------------------------------------------ #
    # FinBERT aggregation (drop-ambiguous-chunks; neutral fully preserved)
    # ------------------------------------------------------------------ #
    def _aggregate_finbert(self, chunks):
        """Run FinBERT on each chunk and combine into one distribution.

        Returns (agg_distribution, n_decisive, decisive_chunks, breakdown,
        guard_fired) where breakdown is a per-decisive-chunk list of
        (top_label, top_conf, guard_applied) for diagnostics."""
        per_chunk_data = []
        guard_fired_any = False
        for ch in chunks:
            out = self.model(ch, truncation=True, max_length=512, top_k=None)
            scores = out[0] if out and isinstance(out[0], list) else out
            dist = {}
            for s in scores:
                dist[s['label'].lower()] = float(s['score'])
            for k in ('positive', 'negative', 'neutral'):
                dist.setdefault(k, 0.0)
            dist, fired = self._finance_guard_adjust(ch, dist)
            guard_fired_any = guard_fired_any or fired
            top_conf = max(dist.values())
            per_chunk_data.append((ch, dist, top_conf, fired))

        decisive = [(ch, d, fired) for ch, d, conf, fired in per_chunk_data if conf >= 0.50]
        if not decisive:
            best = max(per_chunk_data, key=lambda x: x[2])
            decisive = [(best[0], best[1], best[3])]

        agg = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        total_w = 0.0
        breakdown = []
        for ch, dist, fired in decisive:
            w = float(len(ch)) + 1e-6
            total_w += w
            for k in agg:
                agg[k] += dist[k] * w
            top_lbl = max(dist, key=dist.get)
            breakdown.append((top_lbl, dist[top_lbl], fired))
        for k in agg:
            agg[k] /= total_w

        decisive_chunks = [ch for ch, _, _ in decisive]
        return agg, len(decisive), decisive_chunks, breakdown, guard_fired_any

    def _disambiguate(self, agg):
        """v5: resolve the final label from a distribution.

        If the raw winner is positive/negative but sits within neutral_margin
        of the neutral score, treat it as neutral — a near-tie on financial
        text shouldn't surface as a directional sentiment. Returns
        (label, confidence, downgraded_bool)."""
        raw_label = max(agg, key=agg.get)
        if raw_label != 'neutral':
            if (agg[raw_label] - agg['neutral']) < self.neutral_margin:
                return 'neutral', agg['neutral'], True
        return raw_label, agg[raw_label], False

    @staticmethod
    def _excerpts_from(scored_text, max_excerpts=3):
        if not scored_text:
            return []
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', scored_text) if s.strip()]
        substantive = [s for s in sents if len(s) > 40] or sents
        return substantive[:max_excerpts]

    # ------------------------------------------------------------------ #
    # Sentiment (CB-focused + threshold from settings)
    # ------------------------------------------------------------------ #
    _SKIP_TITLES = {"untitled article", "untitled", ""}

    def analyze_sentiment_with_explanation(self, text, title=""):
        focus_text, focus_mode, cb_mentioned, cb_relevance = self._build_focus_text(text)

        # ---- Stage 1: headline-first (CB-gated) ----
        headline = (title or "").strip()
        headline_is_real = (
            len(headline) > 15
            and headline.lower() not in self._SKIP_TITLES
            and not headline.lower().startswith("error:")
        )
        if headline_is_real and self._headline_mentions_cb(headline):
            title_agg, _, _, _, _ = self._aggregate_finbert([headline])
            title_label, title_conf, title_downgraded = self._disambiguate(title_agg)
            if title_conf >= self.confidence_threshold:
                dg_note = (" Near-tie with neutral — reported as NEUTRAL." if title_downgraded else "")
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
                        f"negative {title_agg['negative']:.0%}.{dg_note}"
                    ),
                    'focus_mode': 'headline',
                    'cb_mentioned': cb_mentioned,
                    'cb_relevance': cb_relevance,
                    'distribution': title_agg,
                    'scored_text': headline,
                    'scored_excerpts': [headline],
                    'chunk_breakdown': [],
                    'finance_guard_fired': False,
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
                'chunk_breakdown': [],
                'finance_guard_fired': False,
                'key_phrases': []
            }

        chunks = self._chunk_text(focus_text)
        agg, n, decisive_chunks, breakdown, guard_fired = self._aggregate_finbert(chunks)

        sentiment, confidence, downgraded = self._disambiguate(agg)
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
        downgrade_note = ""
        if downgraded:
            raw_label = max(agg, key=agg.get)
            downgrade_note = (
                f" Note: raw top class was {raw_label.upper()} ({agg[raw_label]:.0%}) but it "
                f"fell within {self.neutral_margin:.0%} of NEUTRAL ({agg['neutral']:.0%}), "
                f"so it was reported as NEUTRAL (near-tie)."
            )
        guard_note = ""
        if guard_fired:
            guard_note = (
                " Finance guard applied: at least one chunk read NEGATIVE on capital-raise "
                "language (e.g. 'raised … in debt') with no genuine-negative cue, so part of "
                "that negative mass was shifted to neutral."
            )

        explanation = (
            f"FinBERT classified this as {sentiment.upper()} "
            f"(confidence {confidence:.1%}), scored on {focus_label} "
            f"across {n} decisive chunk(s). "
            f"Class distribution — positive {agg['positive']:.0%}, "
            f"neutral {agg['neutral']:.0%}, negative {agg['negative']:.0%}."
            f"{downgrade_note}{guard_note}{review_note}"
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
            'chunk_breakdown': breakdown,
            'finance_guard_fired': guard_fired,
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
    # Shared analysis core — runs the SAME pipeline for URLs and pasted text
    # ------------------------------------------------------------------ #
    def _analyze_content(self, url, title, text, key_sentences, headline_for_scoring=None):
        """Run sentiment, alignment, outcomes and coverage on already-extracted
        content. `url` is used only for coverage-type domain checks (may be
        empty for pasted text). `headline_for_scoring` controls the headline-
        first stage: pass the real headline to enable it, or "" to skip it
        (used by pasted text with no genuine headline)."""
        scoring_title = title if headline_for_scoring is None else headline_for_scoring

        sentiment = self.analyze_sentiment_with_explanation(text, title=scoring_title)
        alignment = self.check_alignment_with_explanation(
            text, cb_mentioned=sentiment.get('cb_mentioned', False)
        )
        outcomes = self.determine_outcomes_with_explanation(
            text, sentiment['label'], alignment['pillars']
        )
        coverage = self.classify_coverage_type(
            url, text, cb_mentioned=sentiment.get('cb_mentioned', False)
        )

        return {
            'url': url, 'title': title, 'status': 'Success',
            'error': False, 'paywall': False, 'corrupted': False,
            'sentiment': sentiment, 'alignment': alignment, 'outcomes': outcomes,
            'coverage': coverage, 'key_sentences': key_sentences[:5],
            'text_preview': text[:300] + "..."
        }

    # ------------------------------------------------------------------ #
    # Orchestration — URL path
    # ------------------------------------------------------------------ #
    def analyze_url(self, url):
        page_data = self.fetch_page_content(url)

        if page_data.get('is_paywall') and not page_data['text']:
            return {
                'url': url, 'title': page_data['title'], 'status': 'Paywall',
                'error': True, 'paywall': True, 'corrupted': False,
                'sentiment': {'label': 'neutral', 'score': 0.0, 'needs_review': True,
                              'cb_mentioned': False, 'cb_relevance': 0, 'focus_mode': 'full',
                              'scored_text': '', 'scored_excerpts': [], 'chunk_breakdown': [],
                              'finance_guard_fired': False,
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
                              'scored_text': '', 'scored_excerpts': [], 'chunk_breakdown': [],
                              'finance_guard_fired': False,
                              'explanation': explanation},
                'alignment': {'status': 'NO', 'score': 0, 'explanation': explanation, 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False, 'explanation': explanation},
                'coverage': {'type': 'Unknown', 'icon': '❓', 'reason': explanation},
                'key_sentences': []
            }

        return self._analyze_content(
            url=url,
            title=page_data['title'],
            text=page_data['text'],
            key_sentences=page_data['key_sentences'],
            headline_for_scoring=None,  # use the extracted title as a real headline
        )

    # ------------------------------------------------------------------ #
    # Orchestration — PASTED TEXT path (v6)
    # ------------------------------------------------------------------ #
    def analyze_text(self, raw_text, headline="", source_url=""):
        """Run the full analysis pipeline on user-pasted article text.

        Mirrors `analyze_url` but skips fetching. `headline` is optional: if
        provided it feeds the CB-gated headline-first stage exactly as a real
        headline would; if blank, scoring uses the body only (the auto display
        title is NOT scored, so it can't short-circuit the result).
        `source_url` is optional and used only for Owned/PR-wire coverage
        detection."""
        text = re.sub(r'\s+', ' ', (raw_text or "")).strip()
        headline = (headline or "").strip()
        source_url = (source_url or "").strip()

        # ---- Too short to analyze: return an error-shaped result ----
        if len(text) < 50:
            explanation = ('Pasted text is too short to analyze (need at least ~50 '
                           'characters of article content).')
            return {
                'url': source_url or '(pasted text)',
                'title': headline or 'Pasted Text (too short)',
                'status': 'Too Short',
                'error': True, 'paywall': False, 'corrupted': False,
                'sentiment': {'label': 'neutral', 'score': 0.0, 'needs_review': True,
                              'cb_mentioned': False, 'cb_relevance': 0, 'focus_mode': 'full',
                              'scored_text': '', 'scored_excerpts': [], 'chunk_breakdown': [],
                              'finance_guard_fired': False, 'explanation': explanation},
                'alignment': {'status': 'NO', 'score': 0, 'explanation': explanation, 'matches': []},
                'outcomes': {'outcomes': [], 'cb_mentioned': False, 'explanation': explanation},
                'coverage': {'type': 'Unknown', 'icon': '❓', 'reason': explanation},
                'key_sentences': []
            }

        # ---- Display title: headline if given, else first words of the body ----
        if headline:
            display_title = self._clean_title(headline) or headline[:120]
        else:
            snippet = text[:80].strip()
            if len(text) > 80:
                snippet = snippet.rsplit(' ', 1)[0] + '…'
            display_title = snippet or 'Pasted Text'

        # ---- Key sentences (context excerpts), same rule as the URL path ----
        sentences = re.split(r'(?<=[.!?])\s+', text)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 60][:10]

        # Cap body length to match the 8k limit used for fetched pages.
        text = text[:8000]

        result = self._analyze_content(
            url=source_url,                  # blank => coverage falls back to text patterns
            title=display_title,
            text=text,
            key_sentences=key_sentences,
            headline_for_scoring=headline,   # "" => headline-first stage is skipped
        )
        # Mark the URL field nicely when no source was supplied.
        if not source_url:
            result['url'] = '(pasted text)'
        return result


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
            body {{
                font-family: Arial, sans-serif;
                margin: 24px;
                line-height: 1.4;
                font-size: 12px;
                color: #222;
            }}
            h1 {{ font-size: 1.5em; color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 6px; margin: 0 0 12px; }}
            h2 {{ font-size: 1.2em; color: #764ba2; margin: 16px 0 6px; }}
            h3 {{ font-size: 1.05em; color: #333; margin: 14px 0 6px; }}
            h4 {{ font-size: 0.95em; margin: 10px 0 4px; }}
            p {{ margin: 4px 0; }}
            ul {{ margin: 4px 0; padding-left: 18px; }}
            li {{ margin: 2px 0; }}
            a {{ word-break: break-all; }}
            .summary {{ background: #f0f2f6; padding: 12px 16px; border-radius: 8px; margin: 12px 0; }}
            .article {{ border: 1px solid #ddd; padding: 12px 16px; margin: 12px 0; border-radius: 8px; page-break-inside: avoid; }}
            .sentiment-positive {{ color: #27ae60; font-weight: bold; }}
            .sentiment-negative {{ color: #e74c3c; font-weight: bold; }}
            .sentiment-neutral {{ color: #f39c12; font-weight: bold; }}
            .alignment-yes {{ color: #27ae60; font-weight: bold; }}
            .alignment-partial {{ color: #f39c12; font-weight: bold; }}
            .alignment-no {{ color: #e74c3c; font-weight: bold; }}
            .explanation {{ background: #f8f9fa; padding: 8px 12px; border-left: 4px solid #667eea; margin: 6px 0; }}
            .scored {{ background: #eef4ff; padding: 8px 12px; border-left: 4px solid #3b6fe0; margin: 6px 0; }}
            .needs-review {{ background: #fff8e1; padding: 8px 12px; border-left: 4px solid #f39c12; margin: 6px 0; }}
            .error-paywall {{ background: #fff3f3; padding: 8px 12px; border-left: 4px solid #e74c3c; margin: 6px 0; }}
            .keyword {{ background: #fff3cd; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            .footer {{ text-align: center; margin-top: 24px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 10px; color: #666; }}
            .badge-review {{ background: #f39c12; color: white; padding: 1px 7px; border-radius: 10px; font-size: 10px; margin-left: 6px; }}
            .badge-paywall {{ background: #e74c3c; color: white; padding: 1px 7px; border-radius: 10px; font-size: 10px; margin-left: 6px; }}
            .badge-cb {{ background: #27ae60; color: white; padding: 1px 7px; border-radius: 10px; font-size: 10px; margin-left: 6px; }}
            .coverage-owned {{ color: #2980b9; font-weight: bold; }}
            .coverage-proactive {{ color: #8e44ad; font-weight: bold; }}
            .coverage-earned {{ color: #27ae60; font-weight: bold; }}
            .coverage-unknown {{ color: #95a5a6; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🏢 Company Image Sentiment Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><em>v8 — pluggable analysis engine (Gemini / Claude / FinBERT) with syndicated-PR
        coverage detection. LLM engines use the same reputational-sentiment prompt as the Claude
        artifact and delegate coverage typing to the rule-based classifier; FinBERT path retains
        v6 pasted-text analysis, v5 near-tie neutral disambiguation, finance false-negative guard,
        per-chunk diagnostics, and word-boundary excerpts.</em></p>

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
            status_label = "CORRUPTED" if result.get('corrupted') else result.get('status', 'FAILED').upper()
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
            'headline': "the article headline",
            'llm': "the LLM's full reading"
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

        # ---- Scored text: exactly what FinBERT read ----
        scored_excerpts = result['sentiment'].get('scored_excerpts', [])
        if scored_excerpts:
            html_content += f"""
            <h4>🔬 Scored Text <span style="font-weight:normal;font-size:0.85em;color:#555;">
            (what the engine actually read — {focus_label})</span></h4>
            <div class="scored">
                <ul>
            """
            for s in scored_excerpts:
                html_content += f"<li>\"{EnhancedCompanyAnalyzer._truncate_words(s, 240)}\"</li>"
            html_content += "</ul></div>"

        # ---- Per-chunk diagnostic (v5) ----
        breakdown = result['sentiment'].get('chunk_breakdown', [])
        if breakdown:
            html_content += "<p style='font-size:0.85em;color:#555;'><strong>Per-chunk read:</strong> "
            html_content += " · ".join(
                f"chunk {i}: {lbl.upper()} {conf:.0%}{' (guard)' if fired else ''}"
                for i, (lbl, conf, fired) in enumerate(breakdown, 1)
            )
            html_content += "</p>"

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
                html_content += f"<li>\"{EnhancedCompanyAnalyzer._truncate_words(sentence, 200)}\"</li>"
            html_content += "</ul>"

        html_content += "</div>"

    html_content += """
        <div class="footer">
            <p>Report generated by Company Image Sentiment Analyzer v8</p>
            <p>Pluggable engine (Gemini / Claude / FinBERT) · syndicated-PR coverage detection · pasted-text analysis · near-tie neutral disambiguation · finance false-negative guard · per-chunk diagnostics · word-boundary excerpts · CB-gated headline-first · scored-text display · CB-focused text · CB mention gate · corrupted-content voiding · paywall detection</p>
        </div>
    </body>
    </html>
    """

    return html_content


# ------------------------------------------------------------------ #
# PDF Report (built directly with fpdf2 — no system deps)
# ------------------------------------------------------------------ #
# fpdf2's core fonts (Helvetica) only cover Latin-1, so we sanitize text:
# replace common smart punctuation and drop emoji / other non-Latin-1 glyphs.
_PDF_PUNCT = {
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
    "\u00a0": " ", "\u2192": "->", "\u00b7": "-", "\u2122": "(TM)",
}

# Sentiment / alignment label colors (R, G, B) matching the HTML report.
_PDF_COLORS = {
    "positive": (39, 174, 96), "negative": (231, 76, 60),
    "neutral": (243, 156, 18),
    "YES": (39, 174, 96), "PARTIAL": (243, 156, 18), "NO": (231, 76, 60),
}


def _pdf_safe(text):
    """Make a string safe for fpdf2 core fonts (Latin-1 only)."""
    if text is None:
        return ""
    text = str(text)
    for bad, good in _PDF_PUNCT.items():
        text = text.replace(bad, good)
    # Drop anything Latin-1 can't represent (emoji, CJK, etc.).
    return text.encode("latin-1", "ignore").decode("latin-1")


def create_pdf_report(results):
    """Build a clean PDF report directly with fpdf2.

    Carries the same content as the HTML report (summary stats, per-article
    sentiment, alignment, outcomes, color-coded labels). Returns PDF bytes,
    or None if fpdf2 is missing / generation fails (caller falls back to HTML).
    """
    if not FPDF_AVAILABLE:
        return None

    try:
        pdf = FPDF(format="letter")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_title("Sentiment Analysis Report")
        pdf.add_page()

        def line(text, size=10, style="", color=(34, 34, 34), height=5):
            pdf.set_font("Helvetica", style, size)
            pdf.set_text_color(*color)
            pdf.multi_cell(0, height, _pdf_safe(text))

        def label_line(prefix, value, value_color=(34, 34, 34), size=10):
            """A bold label followed by a colored value, on one wrapped block."""
            pdf.set_font("Helvetica", "B", size)
            pdf.set_text_color(34, 34, 34)
            pdf.write(5, _pdf_safe(prefix + " "))
            pdf.set_font("Helvetica", "", size)
            pdf.set_text_color(*value_color)
            pdf.write(5, _pdf_safe(value))
            pdf.ln(6)

        # ---- Header ----
        line("Company Image Sentiment Analysis Report", size=16, style="B",
             color=(102, 126, 234), height=8)
        line("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             size=9, color=(90, 90, 90))
        pdf.ln(2)

        # ---- Executive summary ----
        successful = [r for r in results if not r.get('error', False)]
        needs_review = [r for r in successful if r['sentiment'].get('needs_review', False)]
        paywalled = [r for r in results if r.get('paywall', False)]
        corrupted = [r for r in results if r.get('corrupted', False)]
        failed = [r for r in results if r.get('status') == 'Failed']
        cb_articles = [r for r in successful if r['outcomes'].get('cb_mentioned', False)]

        line("Executive Summary", size=13, style="B", color=(118, 75, 162), height=7)
        line(f"Total articles: {len(results)}    "
             f"Successfully analyzed: {len(successful)}", size=10)
        line(f"Needs manual review: {len(needs_review)}    "
             f"Paywalled: {len(paywalled)}    "
             f"Corrupted/Failed: {len(corrupted) + len(failed)}", size=10)
        line(f"CB-mentioned articles: {len(cb_articles)}", size=10)

        if successful:
            pos = sum(1 for r in successful if r['sentiment']['label'] == 'positive')
            neg = sum(1 for r in successful if r['sentiment']['label'] == 'negative')
            neu = sum(1 for r in successful if r['sentiment']['label'] == 'neutral')
            aligned = sum(1 for r in successful if r['alignment']['status'] == 'YES')
            owned = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Owned')
            proactive = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Proactive')
            earned = sum(1 for r in results if r.get('coverage', {}).get('type') == 'Earned')
            line(f"Sentiment - Positive: {pos}   Negative: {neg}   Neutral: {neu}", size=10)
            line(f"Strong alignment: {aligned}", size=10)
            line(f"Coverage - Owned: {owned}   Proactive: {proactive}   Earned: {earned}", size=10)
        pdf.ln(3)

        # ---- Per-article detail ----
        for idx, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')

            # Article heading
            line(f"{idx}. {title}", size=11, style="B", color=(51, 51, 51), height=6)
            line("URL: " + result.get('url', ''), size=8, color=(90, 90, 110))

            if result.get('paywall'):
                line("PAYWALL - manual entry required after reading the article directly.",
                     size=9, color=(231, 76, 60))
                pdf.ln(3)
                continue

            if result.get('error', False):
                status_label = "CORRUPTED" if result.get('corrupted') else result.get('status', 'FAILED').upper()
                line(f"Status: {status_label}", size=9, color=(231, 76, 60))
                line(result['sentiment'].get('explanation', ''), size=9, color=(90, 90, 90))
                pdf.ln(3)
                continue

            coverage = result.get('coverage', {'type': 'Unknown', 'reason': ''})
            label_line("Coverage type:", coverage.get('type', 'Unknown').upper(), size=9)
            line("  " + coverage.get('reason', ''), size=8, color=(90, 90, 90))

            sent = result['sentiment']
            sent_color = _PDF_COLORS.get(sent['label'], (34, 34, 34))
            label_line("Sentiment:",
                       f"{sent['label'].upper()}  (confidence {sent['score']:.1%})"
                       + ("  [NEEDS REVIEW]" if sent.get('needs_review') else ""),
                       value_color=sent_color, size=10)
            line("Explanation: " + sent.get('explanation', ''), size=8, color=(70, 70, 70))

            # Scored excerpts
            scored = sent.get('scored_excerpts', [])
            if scored:
                line("Scored text (what the engine read):", size=9, style="B", color=(59, 111, 224))
                for s in scored:
                    line("  - " + EnhancedCompanyAnalyzer._truncate_words(s, 240),
                         size=8, color=(60, 60, 60))

            align = result['alignment']
            align_color = _PDF_COLORS.get(align['status'], (34, 34, 34))
            label_line("Positioning alignment:",
                       f"{align['status']}  (score {align.get('score', 0)}/100  |  "
                       f"CB relevance {sent.get('cb_relevance', 0)}/100)",
                       value_color=align_color, size=10)
            line("Explanation: " + align.get('explanation', ''), size=8, color=(70, 70, 70))

            outc = result['outcomes']
            cb_flag = outc.get('cb_mentioned', False)
            label_line("CB mentioned:", "Yes" if cb_flag else "No - outcomes not tagged",
                       value_color=(39, 174, 96) if cb_flag else (149, 165, 166), size=9)
            names = ', '.join(o['outcome'] for o in outc['outcomes']) if outc['outcomes'] else 'None'
            line("Business outcomes: " + names, size=9, color=(70, 70, 70))

            pdf.ln(2)
            pdf.set_draw_color(220, 220, 220)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)

        # ---- Footer note ----
        pdf.ln(2)
        line("Report generated by Company Image Sentiment Analyzer v8", size=8, color=(120, 120, 120))

        out = pdf.output()  # fpdf2 >= 2.x returns a bytearray
        return bytes(out)

    except Exception:
        return None


# ------------------------------------------------------------------ #
# Export buttons (shared by both input modes)
# ------------------------------------------------------------------ #
def render_exports(results):
    st.markdown("---")
    st.markdown("### 📄 Export Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        df = pd.DataFrame([{
            'URL': r['url'],
            'Title': r['title'],
            'Status': r['status'],
            'Sentiment': r['sentiment']['label'].upper() if not r.get('error') else 'ERROR',
            'Confidence': f"{r['sentiment']['score']:.1%}" if not r.get('error') else 'N/A',
            'Needs Review': r['sentiment'].get('needs_review', True) if not r.get('error') else True,
            'Focus Mode': r['sentiment'].get('focus_mode', 'full') if not r.get('error') else 'N/A',
            'Finance Guard': r['sentiment'].get('finance_guard_fired', False) if not r.get('error') else False,
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
        st.caption("💡 Or open the HTML and Print → Save as PDF")

    with col4:
        if FPDF_AVAILABLE:
            pdf_bytes = create_pdf_report(results)
            if pdf_bytes:
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button(
                    "📥 Download PDF", disabled=True, use_container_width=True,
                    help="PDF generation failed for this batch — use the HTML report instead."
                )
        else:
            st.button(
                "📥 Download PDF", disabled=True, use_container_width=True,
                help="Add `fpdf2` to requirements.txt to enable direct PDF export."
            )
        st.caption("📄 Native PDF (emojis omitted for clean print)")


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
        # ============================================================== #
        # v7: choose which engine does the judgment
        # ============================================================== #
        st.markdown("### 🧠 Analysis Engine")
        engine_choice = st.radio(
            "Analysis engine",
            ["Gemini (free tier)",
             "Claude (paid, matches your artifact)",
             "FinBERT (local, no API)"],
            help="Gemini = a free LLM, $0 for low volume. "
                 "Claude = same quality as your artifact, billed to your key. "
                 "FinBERT = fully local, no API at all.",
        )
        st.caption(
            "The LLM engines (Gemini / Claude) read the whole article and judge "
            "reputational impact — they ignore the FinBERT tuning sliders below."
        )

        st.markdown("---")
        st.markdown("### ⚙️ Tuning Controls")
        st.caption("These apply to the FinBERT engine only.")

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
        neutral_margin = st.slider(
            "Near-tie → neutral margin",
            min_value=0.00, max_value=0.20, value=0.10, step=0.01,
            help="If POSITIVE or NEGATIVE wins but is within this margin of the NEUTRAL "
                 "score, the result is reported as NEUTRAL. Stops FinBERT near-ties on "
                 "dry financial text from surfacing as directional sentiment. Set to 0 to disable."
        )
        finance_guard = st.toggle(
            "Finance false-negative guard",
            value=True,
            help="Nudges a chunk that FinBERT reads NEGATIVE purely on capital-raise "
                 "language ('raised … in debt', 'secured', 'record') toward neutral, "
                 "but ONLY when no genuine-negative word (loss, default, lawsuit…) is present. "
                 "Lexical override — monitor against your labeled set."
        )

        max_workers = st.slider("Parallel workers", 1, 3, 1)
        st.caption("Keep at 1 on Streamlit Community Cloud (1 GB RAM) to avoid out-of-memory crashes. "
                   "On the LLM engines you can raise this, but watch the provider's per-minute rate limit.")

        st.markdown("---")
        st.markdown("### 📊 How the tuning works")
        st.info(
            "**Headline-first (CB-gated):** if the headline NAMES CrossBoundary and "
            "FinBERT is confident on it, that's the score. Generic topic headlines "
            "no longer override CB sentiment.\n\n"
            "**Drop ambiguous chunks:** chunks where no class reaches 50% are excluded "
            "from averaging — they add noise, not signal. Neutral can still win.\n\n"
            "**Near-tie → neutral:** a barely-negative-over-neutral split is reported "
            "as neutral, not negative.\n\n"
            "**Finance guard:** capital-raise language no longer drags a chunk negative.\n\n"
            "**Syndicated-PR catch:** a CB release reposted verbatim on a third-party "
            "site is classified Proactive, not Earned (applies to all three engines).\n\n"
            "**CB focus:** sentiment reflects how the article feels about CrossBoundary.\n\n"
            "**Scored text shown:** the report displays the exact text the engine read."
        )

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info(
            "- Expand any result for the full explanation\n"
            "- Check '🔬 Scored Text' to see exactly what the engine read\n"
            "- Check the per-chunk read to see which chunk drove the score (FinBERT)\n"
            "- Download CSV and filter 'Needs Review = TRUE' first\n"
            "- Lower the threshold if too many true-positive CB articles get flagged\n"
            "- **Paste-text mode:** add a Headline only if the source actually has one\n"
            "- **Privacy:** the free tier may train on your prompts — send public article text only"
        )

    analyzer = st.session_state.analyzer
    analyzer.confidence_threshold = confidence_threshold
    analyzer.focus_cb = focus_cb
    analyzer.neutral_margin = neutral_margin
    analyzer.finance_guard = finance_guard

    # ============================================================== #
    # v7: pick the active engine. All three share analyze_url() /
    # analyze_text() and return the SAME result shape, so everything
    # downstream (exports, UI, coverage) is unchanged. The LLM engines
    # delegate coverage typing to analyzer.classify_coverage_type(), so the
    # v8 syndicated-PR catch applies to every engine automatically.
    # ============================================================== #
    if engine_choice.startswith("Gemini"):
        engine = FreeAnalyzer(helper=analyzer, provider="Gemini (Google AI Studio)")
        engine.confidence_threshold = confidence_threshold
    elif engine_choice.startswith("Claude"):
        engine = ClaudeAnalyzer(helper=analyzer, model="claude-sonnet-4-6")
        engine.confidence_threshold = confidence_threshold
    else:
        engine = analyzer

    # ---- Top-level input mode: URLs vs pasted text ----
    st.markdown("### 🧭 Choose what to analyze")
    source_type = st.radio(
        "Input source:",
        ["Analyze URLs", "Analyze pasted text"],
        horizontal=True,
        help="URLs are fetched and scraped. Pasted text is analyzed directly with "
             "the exact same pipeline (useful for paywalled, "
             "PDF, newsletter, or print coverage you already have in hand)."
    )

    # ============================================================== #
    # MODE 1 — URLs (unchanged behavior)
    # ============================================================== #
    if source_type == "Analyze URLs":
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
                futures = {executor.submit(engine.analyze_url, url): url for url in urls}
                for idx, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(urls))
                    status_text.text(f"Analyzed {idx + 1}/{len(urls)} URLs")

            st.session_state.results = results

            successful = [r for r in results if not r.get('error', False)]
            st.success(f"✅ Analysis complete! {len(successful)}/{len(results)} URLs successfully analyzed")

            display_detailed_results(results)
            render_exports(results)

        elif analyze_btn and not urls:
            st.warning("⚠️ Please enter at least one URL to analyze")

        elif 'results' in st.session_state and st.session_state.results:
            if st.button("📊 Show Previous Results"):
                display_detailed_results(st.session_state.results)

    # ============================================================== #
    # MODE 2 — Pasted text (v6, same pipeline)
    # ============================================================== #
    else:
        st.markdown("### 📋 Paste Article Text to Analyze")
        st.caption(
            "Paste the full article body below. The text runs through the exact same "
            "pipeline used for URLs — sentiment, "
            "positioning alignment, business outcomes, and coverage type."
        )

        pasted_text = st.text_area(
            "Article text:",
            height=300,
            placeholder="Paste the article content here (at least a paragraph or two)..."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            pasted_headline = st.text_input(
                "Headline (optional)",
                placeholder="Leave blank if the source has no real headline",
                help="If you provide a headline that NAMES CrossBoundary, the CB-gated "
                     "headline-first stage runs exactly as it does for URLs. If left "
                     "blank, scoring uses the body text only."
            )
        with col_b:
            pasted_source_url = st.text_input(
                "Source URL (optional)",
                placeholder="e.g. https://prnewswire.com/... or a CB-owned domain",
                help="Used only to classify coverage type (Owned / PR-wire). "
                     "Leave blank to let coverage type be inferred from the text alone."
            )

        analyze_text_btn = st.button("🔍 Analyze Pasted Text", type="primary", use_container_width=True)

        if analyze_text_btn and pasted_text.strip():
            with st.spinner("Analyzing pasted text..."):
                result = engine.analyze_text(
                    raw_text=pasted_text,
                    headline=pasted_headline,
                    source_url=pasted_source_url,
                )
            results = [result]
            st.session_state.results = results

            if result.get('error'):
                st.warning(f"⚠️ {result['status']}: {result['sentiment']['explanation']}")
            else:
                st.success("✅ Analysis complete!")

            display_detailed_results(results)
            render_exports(results)

        elif analyze_text_btn and not pasted_text.strip():
            st.warning("⚠️ Please paste some article text to analyze")

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
                else:
                    st.info(result['sentiment'].get('explanation', ''))
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

            # ---- Scored text: exactly what the engine read ----
            scored_excerpts = result['sentiment'].get('scored_excerpts', [])
            focus_label = {
                'cb_context': "CrossBoundary-specific sentences",
                'pillar': "topic-relevant sentences",
                'full': "the full article body",
                'headline': "the article headline",
                'llm': "the LLM's full reading"
            }.get(focus_mode, focus_mode)
            if scored_excerpts:
                st.markdown(f"#### 🔬 Scored Text — what the engine actually read ({focus_label})")
                for i, s in enumerate(scored_excerpts, 1):
                    st.markdown(
                        f"<div class='scored-box'><strong>{i}.</strong> \"{EnhancedCompanyAnalyzer._truncate_words(s, 280)}\"</div>",
                        unsafe_allow_html=True
                    )

            # ---- Per-chunk diagnostic (v5, FinBERT only) ----
            breakdown = result['sentiment'].get('chunk_breakdown', [])
            if breakdown:
                chunk_str = " · ".join(
                    f"chunk {i}: **{lbl.upper()}** {conf:.0%}{' _(guard)_' if fired else ''}"
                    for i, (lbl, conf, fired) in enumerate(breakdown, 1)
                )
                st.caption(f"Per-chunk read → {chunk_str}")

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
                        f"<div class='explanation-box'><strong>{i}.</strong> \"{EnhancedCompanyAnalyzer._truncate_words(sentence, 250)}\"</div>",
                        unsafe_allow_html=True
                    )


if __name__ == "__main__":
    main()
