import re
from youtube_transcript_api import YouTubeTranscriptApi


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def get_transcript(url: str) -> tuple[str, str]:
    """Return (full_text, video_id). Raises ValueError on bad URL."""
    video_id = _extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    entries = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join(e["text"] for e in entries)
    return text, video_id
