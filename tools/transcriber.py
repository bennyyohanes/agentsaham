"""
Transcriber tool for AgentSaham.

Supports transcribing audio/video files (local and YouTube URLs) using
faster-whisper with automatic language detection for Indonesian and English.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}

# Allowlist of valid YouTube URL patterns
_YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/(watch\?.*v=[\w-]+|shorts/[\w-]+)|youtu\.be/[\w-]+)"
)


def _is_youtube_url(source: str) -> bool:
    """Check if a string is a valid YouTube URL using a strict allowlist regex."""
    return bool(_YOUTUBE_URL_RE.match(source))


def _validate_youtube_url(url: str) -> str:
    """
    Validate and return a sanitized YouTube URL.

    Args:
        url: URL string to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL does not match the YouTube URL allowlist.
    """
    if not _YOUTUBE_URL_RE.match(url):
        raise ValueError(
            f"Invalid YouTube URL: '{url}'. "
            "Only youtube.com/watch, youtube.com/shorts, and youtu.be URLs are supported."
        )
    return url


def _download_youtube_audio(url: str, output_path: str) -> str:
    """
    Download audio from a YouTube video using yt-dlp.

    Args:
        url: Validated YouTube video URL.
        output_path: Output file path (without extension; yt-dlp adds .m4a/.opus etc.).
                     Must be within a safe temporary directory.

    Returns:
        Path to the downloaded audio file.

    Raises:
        ValueError: If the URL fails validation.
        RuntimeError: If yt-dlp fails.
    """
    # Re-validate URL before passing to subprocess
    validated_url = _validate_youtube_url(url)

    # Ensure output_path is inside a temp directory (resolved absolute path)
    output_resolved = str(Path(output_path).resolve())

    logger.info(f"Downloading YouTube audio from validated URL")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_resolved + ".%(ext)s",
        "--",      # Treat subsequent arguments as non-option strings
        validated_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file
    parent = Path(output_resolved).parent
    stem = Path(output_resolved).stem
    for f in parent.iterdir():
        if f.stem == stem and f.suffix in (".mp3", ".m4a", ".opus", ".wav"):
            return str(f)
    raise RuntimeError("yt-dlp finished but output file not found")


def _extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """
    Extract audio stream from a video file using ffmpeg.

    Args:
        video_path: Resolved absolute path to the input video file.
        audio_path: Resolved absolute path to write the output audio file (.wav).

    Raises:
        RuntimeError: If ffmpeg is not available or extraction fails.
    """
    # Use resolved absolute paths to prevent path traversal
    resolved_video = str(Path(video_path).resolve())
    resolved_audio = str(Path(audio_path).resolve())

    logger.debug(f"Extracting audio from video file")
    cmd = [
        "ffmpeg",
        "-y",          # Overwrite output
        "-i", resolved_video,
        "-vn",         # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        resolved_audio,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")


class Transcriber:
    """
    Transcribes audio/video content using faster-whisper.

    Supports:
    - Local video files (.mp4, .avi, .mkv, etc.)
    - YouTube URLs (via yt-dlp)
    - Automatic language detection (Indonesian / English)

    Attributes:
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        device: Compute device ("cpu" or "cuda").
        compute_type: Quantization type ("int8", "float16", "float32").
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """
        Initialize the Transcriber and load the Whisper model.

        Args:
            model_size: Whisper model variant. Defaults to "small" for CPU use.
            device: Inference device. Use "cuda" if GPU is available.
            compute_type: Quantization mode for faster inference.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        """Lazy-load the faster-whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Run: pip install faster-whisper"
                ) from exc

            logger.info(
                f"Loading Whisper model: {self.model_size} "
                f"(device={self.device}, compute={self.compute_type})"
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, source: str) -> dict:
        """
        Transcribe an audio/video source.

        Args:
            source: Either a local file path or a YouTube URL.

        Returns:
            Dict with keys:
                - "text": Full transcription string.
                - "segments": List of dicts with {start, end, text}.
                - "language": Detected language code (e.g. "id", "en").
                - "source": Original source string.

        Raises:
            ValueError: If the file extension is not supported.
            RuntimeError: If ffmpeg or yt-dlp encounters an error.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = self._prepare_audio(source, tmp_dir)
            return self._run_transcription(audio_path, source)

    def _prepare_audio(self, source: str, tmp_dir: str) -> str:
        """
        Prepare audio file from source (URL or local video).

        Validates the source: YouTube URLs are validated via allowlist regex;
        local file paths are resolved to absolute paths to prevent traversal.
        """
        if _is_youtube_url(source):
            # URL is validated inside _download_youtube_audio
            base_path = os.path.join(tmp_dir, "yt_audio")
            audio_file = _download_youtube_audio(source, base_path)
            return audio_file

        # Resolve to absolute path to prevent path traversal attacks
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        ext = path.suffix.lower()
        all_supported = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
        if ext not in all_supported:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {sorted(all_supported)}"
            )

        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            audio_path = str(Path(tmp_dir).resolve() / "audio.wav")
            _extract_audio_from_video(str(path), audio_path)
            return audio_path

        # Already an audio file — return the resolved path string
        return str(path)

    def _run_transcription(self, audio_path: str, original_source: str) -> dict:
        """Run faster-whisper transcription on a prepared audio file."""
        model = self._load_model()
        logger.info(f"Transcribing audio file: {audio_path}")

        segments, info = model.transcribe(
            audio_path,
            language=None,  # Auto-detect
            task="transcribe",
            beam_size=5,
        )

        detected_language = info.language
        logger.info(f"Detected language: {detected_language} (probability={info.language_probability:.2f})")

        segment_list = []
        full_text_parts = []
        for seg in segments:
            segment_list.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)
        logger.info(f"Transcription complete: {len(segment_list)} segments, {len(full_text)} chars")

        return {
            "text": full_text,
            "segments": segment_list,
            "language": detected_language,
            "source": original_source,
        }
