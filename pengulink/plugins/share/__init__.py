"""pengulink.plugins.share
~~~~~~~~~~~~~~~~~~~~~~~
KDE Connect Share plugin ecosystem for PenguLink daemon.
"""

from pengulink.plugins.share.security import (
    AtomicFileStager,
    AtomicStagingWriter,
    StagingFileWriter,
    cleanup_orphaned_part_files,
    resolve_collision,
    resolve_download_dir,
    sanitize_filename,
    strip_execution_bits,
    validate_containment,
    verify_boundary,
)
from pengulink.plugins.share.payload import (
    BatchMetadata,
    PayloadClient,
    PayloadServer,
    PayloadTransferManager,
    TransferProgress,
)
from pengulink.plugins.share.plugin import (
    SHARE_LEGACY_TYPE,
    SHARE_REQUEST_TYPE,
    SHARE_UPDATE_TYPE,
    SharePlugin,
    create_share_file_packet,
    create_share_text_packet,
    create_share_url_packet,
)

__all__ = [
    "SharePlugin",
    "SHARE_REQUEST_TYPE",
    "SHARE_UPDATE_TYPE",
    "SHARE_LEGACY_TYPE",
    "create_share_text_packet",
    "create_share_url_packet",
    "create_share_file_packet",
    "AtomicFileStager",
    "AtomicStagingWriter",
    "StagingFileWriter",
    "cleanup_orphaned_part_files",
    "resolve_collision",
    "resolve_download_dir",
    "sanitize_filename",
    "strip_execution_bits",
    "validate_containment",
    "verify_boundary",
    "BatchMetadata",
    "PayloadClient",
    "PayloadServer",
    "PayloadTransferManager",
    "TransferProgress",
]
