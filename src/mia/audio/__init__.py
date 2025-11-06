# Audio module init
# Minimal imports to avoid dependency issues during build

__all__ = ["get_audio_utils", "get_speech_processor", "get_speech_generator"]


# Lazy imports to avoid build-time dependency issues
def get_audio_utils():
    from .audio_utils import AudioUtils

    return AudioUtils


def get_speech_processor():
    from .speech_processor import SpeechProcessor

    return SpeechProcessor


def get_speech_generator():
    from .speech_generator import SpeechGenerator

    return SpeechGenerator
