"""Transcript extraction: YouTube captions first, local Whisper fallback."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from brand_guardian.exception import BrandGuardianException
from brand_guardian.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LANGUAGES = ["en"]
WHISPER_MODEL_SIZE = "base"


@dataclass
class TranscriptSegment:
    start: float
    duration: float
    text: str


@dataclass
class TranscriptResult:
    video_id: str
    text: str
    segments: list[TranscriptSegment]
    source: str  # "captions" | "whisper"


def extract_video_id(video_url: str) -> str:
    parsed = urlparse(video_url)
    video_id = ""

    if parsed.hostname == "youtu.be":
        video_id = parsed.path.lstrip("/")
    elif parsed.hostname and "youtube.com" in parsed.hostname:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            # Covers /embed/<id>, /shorts/<id>, /v/<id>
            video_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]

    if not video_id:
        raise BrandGuardianException(ValueError(f"Could not extract a video ID from URL: {video_url}"))
    return video_id


def _fetch_captions(video_id: str, languages: list[str]) -> TranscriptResult | None:
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(languages)
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(languages)

        fetched = transcript.fetch()
        segments = [
            TranscriptSegment(start=snippet.start, duration=snippet.duration, text=snippet.text)
            for snippet in fetched
        ]
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        logger.info(f"Fetched captions for {video_id} (language={transcript.language_code})")
        return TranscriptResult(video_id=video_id, text=text, segments=segments, source="captions")
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.info(f"No usable captions for {video_id}, falling back to Whisper")
        return None
    except Exception as error:
        raise BrandGuardianException(error) from error


@lru_cache(maxsize=1)
def _get_whisper_model(model_size: str = WHISPER_MODEL_SIZE):
    from faster_whisper import WhisperModel

    logger.info(f"Loading local Whisper model '{model_size}'")
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _download_audio(video_url: str, dest_dir: Path) -> Path:
    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        return Path(ydl.prepare_filename(info))


def _transcribe_with_whisper(video_id: str, video_url: str) -> TranscriptResult:
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = _download_audio(video_url, Path(tmp_dir))
            model = _get_whisper_model()
            raw_segments, _info = model.transcribe(str(audio_path))
            segments = [
                TranscriptSegment(start=s.start, duration=s.end - s.start, text=s.text.strip())
                for s in raw_segments
            ]
        text = " ".join(segment.text for segment in segments if segment.text)
        logger.info(f"Transcribed {video_id} locally with Whisper ({len(segments)} segments)")
        return TranscriptResult(video_id=video_id, text=text, segments=segments, source="whisper")
    except Exception as error:
        raise BrandGuardianException(error) from error


def get_transcript(video_url: str, languages: list[str] | None = None) -> TranscriptResult:
    """Get a transcript for a YouTube video: captions first, local Whisper fallback."""
    video_id = extract_video_id(video_url)
    languages = languages or DEFAULT_LANGUAGES

    captions_result = _fetch_captions(video_id, languages)
    if captions_result is not None:
        return captions_result

    return _transcribe_with_whisper(video_id, video_url)
