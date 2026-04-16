import os
import re
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.proxies import WebshareProxyConfig


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _build_transcript_api() -> YouTubeTranscriptApi:
    username = os.getenv("WEBSHARE_PROXY_USERNAME")
    password = os.getenv("WEBSHARE_PROXY_PASSWORD")
    if username and password:
        proxy_config = WebshareProxyConfig(
            proxy_username=username,
            proxy_password=password,
        )
        return YouTubeTranscriptApi(proxy_config=proxy_config)
    return YouTubeTranscriptApi()


def get_youtube_tool():
    search = TavilySearchResults(k=5)
    transcript_api = _build_transcript_api()

    @tool
    def get_youtube_transcript(query: str) -> str:
        """Search YouTube for a video that matches a query and return its transcript alongside the video url."""
        results = search.invoke(f"site:youtube.com/watch {query}")

        video_ids = []
        for result in results:
            url = result.get("url", "")
            video_id = _extract_video_id(url)
            if video_id and video_id not in video_ids:
                video_ids.append(video_id)

        if not video_ids:
            return "No YouTube videos found for this query."

        for video_id in video_ids:
            try:
                transcript = transcript_api.fetch(video_id)
                text = " ".join([snippet.text for snippet in transcript.snippets])
                return f"Transcript for https://youtube.com/watch?v={video_id}:\n\n{text}"
            except (TranscriptsDisabled, NoTranscriptFound):
                continue
            except Exception:
                continue

        return "Found YouTube videos but could not retrieve transcripts for any of them."

    return get_youtube_transcript
