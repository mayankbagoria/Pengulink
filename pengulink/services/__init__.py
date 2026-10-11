"""pengulink.services
~~~~~~~~~~~~~~~~~~
UI-Independent Core Services for PenguLink.
"""

from __future__ import annotations

from pengulink.services.alert import (
    DEFAULT_MAX_ALERT_DURATION,
    INTER_CHIME_DELAY,
    AlertService,
    AlsaAudioBackend,
    AudioBackend,
    MockAudioBackend,
    MockAudioProcess,
    PipeWireAudioBackend,
    PulseAudioAudioBackend,
    SilentAudioProcess,
    SilentFallbackBackend,
    detect_audio_backend,
    ensure_chime_file,
    generate_alert_chime,
)

__all__ = [
    "DEFAULT_MAX_ALERT_DURATION",
    "INTER_CHIME_DELAY",
    "AlertService",
    "AudioBackend",
    "PipeWireAudioBackend",
    "PulseAudioAudioBackend",
    "AlsaAudioBackend",
    "MockAudioBackend",
    "SilentFallbackBackend",
    "MockAudioProcess",
    "SilentAudioProcess",
    "detect_audio_backend",
    "ensure_chime_file",
    "generate_alert_chime",
]
