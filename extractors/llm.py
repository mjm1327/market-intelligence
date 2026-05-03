import json
import os
import re

import anthropic

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_PROMPT = """\
You are an investment research assistant. Analyze the content below and extract every company that is meaningfully discussed (not just mentioned in passing).

For each company return:
- company_name: full name
- ticker: stock ticker if known, otherwise "UNKNOWN"
- sub_theme: one of [ai_infrastructure, edge_inference, biotech_healthtech, ai_applications, ai_other, not_ai]
- sentiment: one of [bullish, bearish, neutral]
- context: 1-2 sentence explanation of why the company was discussed

Return ONLY a valid JSON array. No markdown fences, no commentary.

Example:
[
  {{
    "company_name": "NVIDIA",
    "ticker": "NVDA",
    "sub_theme": "ai_infrastructure",
    "sentiment": "bullish",
    "context": "Highlighted as the primary beneficiary of AI training infrastructure spend."
  }}
]

Content:
{content}
"""

_MAX_CHARS = 14_000


def extract_companies(content: str) -> list[dict]:
    """Call Claude to extract companies from raw text. Returns a list of dicts."""
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS] + "\n...[truncated]"

    client = _get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": _PROMPT.format(content=content)}],
    )

    text = msg.content[0].text.strip()

    # Strip markdown fences if model wraps anyway
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        # Try to find a JSON array anywhere in the response
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []
