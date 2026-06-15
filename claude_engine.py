# claude_engine.py — Claude (LLM) analysis engine for the Coverage Logger
# ----------------------------------------------------------------------
# Drop this file NEXT TO company_sentiment_analyzer.py.
#
# It exposes a ClaudeAnalyzer with the SAME public methods as your FinBERT
# EnhancedCompanyAnalyzer — analyze_url() and analyze_text() — and returns the
# SAME result dictionary shape. So your existing exports, coverage-type
# classification, review queue, and UI all keep working with zero changes.
#
# It uses the SAME reputational-sentiment prompt your Claude artifact
# (coverage.tsx) uses, so the two tools line up instead of fighting.
#
# Setup:
#   1. pip install anthropic        (and add `anthropic` to requirements.txt)
#   2. In Streamlit secrets (Settings -> Secrets, or .streamlit/secrets.toml):
#          ANTHROPIC_API_KEY = "sk-ant-..."
#
# Note: this calls the Anthropic API with YOUR key and bills YOUR account.
# Sonnet 4.6 is $3 / $15 per million input / output tokens, so a single
# article is roughly a cent. It is NOT free the way the in-Claude artifact is.

import json
import re
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Same instruction set as your coverage.tsx artifact, with two extra keys
# (`confidence` and `cb_mentioned`) so we can feed the review queue and the
# coverage/outcome gates the way the FinBERT version does.
CLAUDE_SYSTEM = """You are a media-coverage analyst for CrossBoundary (CB), a firm focused on frontier-market investment, climate finance, and energy access.
Analyze one piece of coverage and return ONLY a JSON object (no markdown, no backticks, no preamble) with exactly these keys:
{
  "outlet": string,              // publication name, "" if unknown
  "date": string,                // publication date as written, "" if unknown
  "summary": string,             // ONE sentence on what this says about CB
  "sentiment": "Positive" | "Neutral" | "Negative",   // for CB's IMAGE/reputation, not just tone
  "sentiment_reason": string,    // one short clause, tied to reputation
  "confidence": number,          // 0.0-1.0, your confidence in the sentiment call
  "positioning": "Yes" | "Partial" | "No",   // Does it reinforce CB core positioning?
  "pillars": string[],           // subset of ["frontier investment","climate finance","energy access"] actually reinforced
  "positioning_reason": string,  // one short clause
  "business_outcomes": string[], // subset of ["BD Support","Investor Narrative","Talent & Reputation","Partnership","Award/Recognition"]
  "cb_mentioned": boolean,       // is CrossBoundary actually named / discussed?
  "quote": string                // one short notable quote pulled verbatim, "" if none
}
Judge sentiment by reputational impact: factual-but-damaging coverage is Negative even if the tone is neutral.
Output must be a single valid JSON object. Keep "quote" short (under 20 words) and escape any double quotes or newlines inside string values so the JSON parses cleanly."""


_PILLAR_KEY = {
    "frontier investment": "frontier_investment",
    "climate finance": "climate_finance",
    "energy access": "energy_access",
}
_SENT_MAP = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
_POS_MAP = {"yes": "YES", "partial": "PARTIAL", "no": "NO"}
_VALID_OUTCOMES = {
    "BD Support", "Investor Narrative", "Talent & Reputation",
    "Partnership", "Award/Recognition",
}


def _balance_json(s):
    """Best-effort repair of a truncated/unbalanced JSON object: close an open
    string and any unclosed brackets so json.loads can finish the parse."""
    out, stack = [], []
    in_str = escape = False
    for ch in s:
        if escape:
            out.append(ch); escape = False; continue
        if ch == "\\":
            out.append(ch); escape = True; continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack:
                stack.pop()
        out.append(ch)
    res = "".join(out)
    if in_str:
        res += '"'                       # close a string the model cut off
    res = re.sub(r",\s*$", "", res)      # drop a dangling trailing comma
    for opener in reversed(stack):       # close any open brackets
        res += "}" if opener == "{" else "]"
    return res


def _coerce_json(raw):
    """Parse model output into a dict, tolerating code fences, prose around the
    JSON, literal control characters, and truncated/unterminated output."""
    if not raw:
        raise ValueError("Empty model response")
    s = raw.replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    # 1) Parse the first complete object; ignore any trailing prose.
    #    strict=False also allows literal newlines/tabs inside strings.
    try:
        obj, _ = json.JSONDecoder(strict=False).raw_decode(s, start)
        return obj
    except Exception:
        pass

    # 2) Repair a truncated/unbalanced object and try once more.
    return json.loads(_balance_json(s[start:]), strict=False)


class ClaudeAnalyzer:
    """LLM analysis engine. Same surface as EnhancedCompanyAnalyzer, but Claude
    does the judgment. Reuses your FinBERT analyzer ONLY for fetching pages,
    detecting CB mentions, and classifying coverage type."""

    def __init__(self, helper, model="claude-sonnet-4-6"):
        # `helper` is your existing EnhancedCompanyAnalyzer instance.
        self.helper = helper
        self.model = model
        self.confidence_threshold = 0.55   # overwritten from the sidebar, like FinBERT

        api_key = None
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            api_key = None
        self.client = (
            anthropic.Anthropic(api_key=api_key)
            if (ANTHROPIC_AVAILABLE and api_key) else None
        )

    # ------------------------------------------------------------------ #
    # Core LLM call
    # ------------------------------------------------------------------ #
    def _ask_claude(self, text, headline, url):
        user = "Coverage text:\n\n" + text.strip()
        if headline:
            user += f"\n\n(Headline: {headline})"
        if url:
            user += f"\n\n(Source URL: {url})"

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(
            b.text for b in msg.content if getattr(b, "type", "") == "text"
        )
        return _coerce_json(raw)

    # ------------------------------------------------------------------ #
    # Map Claude's JSON into the result dict the rest of the app expects
    # ------------------------------------------------------------------ #
    def _map(self, parsed, url, title, text, key_sentences):
        sent = _SENT_MAP.get(str(parsed.get("sentiment", "neutral")).lower(), "neutral")

        try:
            conf = float(parsed.get("confidence", 0.85))
        except (TypeError, ValueError):
            conf = 0.85
        conf = min(max(conf, 0.0), 1.0)

        pos_status = _POS_MAP.get(str(parsed.get("positioning", "No")).lower(), "NO")

        pillar_keys = []
        for p in (parsed.get("pillars") or []):
            if isinstance(p, str) and p.strip().lower() in _PILLAR_KEY:
                pillar_keys.append(_PILLAR_KEY[p.strip().lower()])

        outcomes = [
            {"outcome": o, "keywords": [], "description": "", "confidence": 90}
            for o in (parsed.get("business_outcomes") or []) if o in _VALID_OUTCOMES
        ]

        # CB mention: trust Claude, fall back to your regex detector.
        cb_mentioned = bool(parsed.get("cb_mentioned", False)) or self.helper._is_cb_mentioned(text)

        summary = (parsed.get("summary") or "").strip()
        quote = (parsed.get("quote") or "").strip()
        sent_reason = (parsed.get("sentiment_reason") or "").strip()
        pos_reason = (parsed.get("positioning_reason") or "").strip()
        outlet = (parsed.get("outlet") or "").strip()
        date = (parsed.get("date") or "").strip()

        coverage = self.helper.classify_coverage_type(url, text, cb_mentioned=cb_mentioned)

        scored_excerpts = [s for s in (summary, quote) if s]

        explanation = (
            f"Claude ({self.model}) judged reputational sentiment as {sent.upper()}"
            + (f" — {sent_reason}" if sent_reason else "")
            + f". Confidence {conf:.0%}."
        )
        if conf < self.confidence_threshold:
            explanation += (
                f" ⚠️ Below {self.confidence_threshold:.0%} threshold — recommend manual review."
            )

        return {
            "url": url or "(pasted text)",
            "title": title,
            "outlet": outlet,
            "date": date,
            "summary": summary,
            "quote": quote,
            "status": "Success",
            "error": False, "paywall": False, "corrupted": False,
            "sentiment": {
                "label": sent,
                "score": conf,
                "needs_review": conf < self.confidence_threshold,
                "explanation": explanation,
                "focus_mode": "llm",
                "cb_mentioned": cb_mentioned,
                "cb_relevance": 100 if cb_mentioned else 0,
                "distribution": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "scored_text": " ".join(scored_excerpts),
                "scored_excerpts": scored_excerpts,
                "chunk_breakdown": [],
                "finance_guard_fired": False,
                "key_phrases": [],
            },
            "alignment": {
                "status": pos_status,
                "score": {"YES": 90, "PARTIAL": 55, "NO": 10}[pos_status],
                "pillars": pillar_keys,
                "explanation": pos_status + (f" — {pos_reason}" if pos_reason else ""),
                "matches": [
                    {"pillar": k, "keywords": [], "description": ""} for k in pillar_keys
                ],
            },
            "outcomes": {
                "outcomes": outcomes,
                "cb_mentioned": cb_mentioned,
                "explanation": summary or "Analyzed by Claude.",
            },
            "coverage": coverage,
            "key_sentences": key_sentences[:5],
            "text_preview": text[:300] + "...",
        }

    @staticmethod
    def _error_result(url, title, status, explanation,
                      paywall=False, corrupted=False):
        """Same error shape the FinBERT app produces, so the UI handles it."""
        return {
            "url": url, "title": title, "status": status,
            "error": True, "paywall": paywall, "corrupted": corrupted,
            "outlet": "", "date": "", "summary": "", "quote": "",
            "sentiment": {"label": "neutral", "score": 0.0, "needs_review": True,
                          "cb_mentioned": False, "cb_relevance": 0, "focus_mode": "llm",
                          "scored_text": "", "scored_excerpts": [], "chunk_breakdown": [],
                          "finance_guard_fired": False, "key_phrases": [],
                          "distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                          "explanation": explanation},
            "alignment": {"status": "NO", "score": 0, "pillars": [],
                          "explanation": explanation, "matches": []},
            "outcomes": {"outcomes": [], "cb_mentioned": False, "explanation": explanation},
            "coverage": {"type": "Unknown", "icon": "❓", "reason": explanation},
            "key_sentences": [],
        }

    # ------------------------------------------------------------------ #
    # URL path — reuse your scraper, then let Claude judge the text
    # ------------------------------------------------------------------ #
    def analyze_url(self, url):
        page = self.helper.fetch_page_content(url)

        if page.get("is_paywall") and not page["text"]:
            return self._error_result(url, page["title"], "Paywall",
                                      "Paywalled article — manual entry required",
                                      paywall=True)
        if page.get("corrupted") or not page["text"]:
            corrupted = page.get("corrupted", False)
            return self._error_result(
                url, page["title"], "Corrupted" if corrupted else "Failed",
                "Corrupted binary content detected — result voided." if corrupted
                else "Could not fetch content.",
                corrupted=corrupted)

        if self.client is None:
            return self._error_result(
                url, page["title"], "No API key",
                "ANTHROPIC_API_KEY missing — add it to Streamlit secrets, or switch to FinBERT.")

        try:
            parsed = self._ask_claude(page["text"], page["title"], url)
        except Exception as e:
            return self._error_result(url, page["title"], "LLM error",
                                      f"Claude analysis failed: {e}")

        return self._map(parsed, url, page["title"], page["text"], page["key_sentences"])

    # ------------------------------------------------------------------ #
    # Pasted-text path
    # ------------------------------------------------------------------ #
    def analyze_text(self, raw_text, headline="", source_url=""):
        text = re.sub(r"\s+", " ", (raw_text or "")).strip()
        headline = (headline or "").strip()
        source_url = (source_url or "").strip()

        if len(text) < 50:
            return self._error_result(
                source_url or "(pasted text)", headline or "Pasted Text (too short)",
                "Too Short",
                "Pasted text is too short to analyze (need at least ~50 characters).")

        if self.client is None:
            return self._error_result(
                source_url or "(pasted text)", headline or "Pasted Text", "No API key",
                "ANTHROPIC_API_KEY missing — add it to Streamlit secrets, or switch to FinBERT.")

        if headline:
            display_title = headline
        else:
            display_title = (text[:80].rsplit(" ", 1)[0] + "…") if len(text) > 80 else text

        sentences = re.split(r"(?<=[.!?])\s+", text)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 60][:10]
        text = text[:12000]

        try:
            parsed = self._ask_claude(text, headline, source_url)
        except Exception as e:
            return self._error_result(source_url or "(pasted text)", display_title,
                                      "LLM error", f"Claude analysis failed: {e}")

        return self._map(parsed, source_url, display_title, text, key_sentences)
