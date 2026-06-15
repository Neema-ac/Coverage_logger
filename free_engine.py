# free_engine.py — free / low-cost LLM engine for the Coverage Logger
# ----------------------------------------------------------------------
# Drop this NEXT TO company_sentiment_analyzer.py and claude_engine.py.
#
# It subclasses ClaudeAnalyzer, so it reuses ALL the same result-mapping,
# coverage typing, error handling, and analyze_url/analyze_text logic. The
# only thing it changes is the model call: instead of the Anthropic SDK it
# uses the OpenAI SDK pointed at a free, OpenAI-compatible endpoint.
#
# Works with three free providers — switch by changing `provider`:
#   - "Gemini (Google AI Studio)"  <- recommended for this task
#   - "Groq (Llama 3.3 70B)"
#   - "OpenRouter"
#
# Setup:
#   1. pip install openai          (add `openai` to requirements.txt)
#   2. Get a free key:
#        Gemini    -> https://aistudio.google.com   (no credit card)
#        Groq      -> https://console.groq.com
#        OpenRouter-> https://openrouter.ai
#   3. In Streamlit secrets, set the matching key:
#        GEMINI_API_KEY     = "..."
#        GROQ_API_KEY       = "..."
#        OPENROUTER_API_KEY = "..."
#
# PRIVACY NOTE: free tiers may use your prompts to train models. Send public
# article text only — never internal/confidential CrossBoundary material.

import json
import re
import streamlit as st

from claude_engine import ClaudeAnalyzer, CLAUDE_SYSTEM

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False


# Each provider exposes an OpenAI-compatible endpoint, so one client works for
# all of them. Model IDs and free limits change — verify on the provider site.
PROVIDERS = {
    "Gemini (Google AI Studio)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "secret": "GEMINI_API_KEY",
    },
    "Groq (Llama 3.3 70B)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "secret": "GROQ_API_KEY",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "secret": "OPENROUTER_API_KEY",
    },
}


class FreeAnalyzer(ClaudeAnalyzer):
    """Same interface and output as ClaudeAnalyzer, but calls a free LLM."""

    def __init__(self, helper, provider="Gemini (Google AI Studio)", model=None):
        self.helper = helper
        self.confidence_threshold = 0.55
        self.provider = provider

        cfg = PROVIDERS[provider]
        self.model = model or cfg["model"]

        key = None
        try:
            key = st.secrets.get(cfg["secret"])
        except Exception:
            key = None

        self.client = (
            OpenAI(base_url=cfg["base_url"], api_key=key)
            if (OPENAI_SDK_AVAILABLE and key) else None
        )

    # Override ONLY the model call. analyze_url(), analyze_text(), _map(),
    # and _error_result() are all inherited from ClaudeAnalyzer unchanged.
    def _ask_claude(self, text, headline, url):
        user = "Coverage text:\n\n" + text.strip()
        if headline:
            user += f"\n\n(Headline: {headline})"
        if url:
            user += f"\n\n(Source URL: {url})"

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": CLAUDE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        raw = (resp.choices[0].message.content or "")
        raw = raw.replace("```json", "").replace("```", "").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        return json.loads(m.group(0) if m else raw)
