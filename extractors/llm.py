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


_EXTRACT_PROMPT = """\
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

_SUMMARIZE_PROMPT = """\
You are an investment research assistant. The following is a transcript from a video or article.

Write a concise investment-focused summary covering:
1. **Main thesis / key argument** (2-3 sentences)
2. **Macro themes discussed** (bullet list)
3. **Investment opportunities highlighted** (bullet list — sectors, trends, or specific plays)
4. **Risks or concerns raised** (bullet list, if any)
5. **Overall sentiment** (bullish / bearish / neutral, with one sentence explanation)

Be direct and specific. Skip filler and focus on what is actionable for an investor.

Transcript:
{content}
"""

_MAX_CHARS = 14_000
_MAX_CHARS_SUMMARY = 20_000  # summaries can handle more context


def extract_companies(content: str) -> list[dict]:
    """Call Claude to extract companies from raw text. Returns a list of dicts."""
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS] + "\n...[truncated]"

    client = _get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(content=content)}],
    )

    text = msg.content[0].text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


def summarize_transcript(content: str) -> str:
    """Generate an investment-focused summary of a transcript."""
    if len(content) > _MAX_CHARS_SUMMARY:
        content = content[:_MAX_CHARS_SUMMARY] + "\n...[truncated]"

    client = _get_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": _SUMMARIZE_PROMPT.format(content=content)}],
    )
    return msg.content[0].text.strip()


def parse_transcript_file(file_bytes: bytes, filename: str) -> str:
    """Parse uploaded transcript files (.txt, .srt, .vtt) into plain text."""
    raw = file_bytes.decode("utf-8", errors="replace")

    if filename.endswith(".srt"):
        return _strip_srt(raw)
    if filename.endswith(".vtt"):
        return _strip_vtt(raw)
    return raw  # .txt or anything else


def _strip_srt(text: str) -> str:
    """Remove SRT sequence numbers and timestamps, return clean dialogue."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}", line):
            continue
        lines.append(line)
    return " ".join(lines)


def _strip_vtt(text: str) -> str:
    """Remove WebVTT headers and timestamps, return clean dialogue."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if re.match(r"[\d:\.]+\s*-->\s*[\d:\.]+", line):
            continue
        lines.append(line)
    return " ".join(lines)
