"""Audio transcription tool using OpenAI Whisper."""

from __future__ import annotations

import os
from typing import Optional

from google.adk.tools import FunctionTool

# Lazy-loaded Whisper model (module-level global)
_whisper_model = None


def audio_transcription(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio using OpenAI Whisper (lazy loaded)."""
    global _whisper_model

    if not os.path.exists(audio_path):
        return f"[ERROR] Audio file not found: {audio_path}"

    try:
        import whisper
    except ImportError:
        return "[ERROR] openai-whisper not installed. Run: pip install openai-whisper"

    try:
        # Lazy load model (base model for speed/accuracy balance)
        if _whisper_model is None:
            print("[INFO] Loading Whisper model (base)...")
            _whisper_model = whisper.load_model("base")

        # Transcribe
        options = {}
        if language:
            options["language"] = language

        result = _whisper_model.transcribe(audio_path, **options)

        transcription = result.get("text", "")

        output = [
            f"Audio file: {os.path.basename(audio_path)}",
            f"Detected language: {result.get('language', 'unknown')}",
            "\nTranscription:",
            transcription,
        ]

        return "\n".join(output)

    except Exception as e:
        return f"[ERROR] Audio transcription failed: {str(e)}"


audio_transcription_tool = FunctionTool(audio_transcription)
