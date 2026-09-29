"""Security utilities: MIME verification, path sanitization, size checks."""

import os
from pathlib import Path
import re
import uuid
from typing import Tuple

# File signature (magic bytes) mappings
MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # followed by WEBP
    b"%PDF": "application/pdf",
    b"II*\x00": "image/tiff",
    b"MM\x00*": "image/tiff",
}


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename against path traversal and special characters."""
    base = os.path.basename(filename)
    # Remove null bytes and path separators
    clean = re.sub(r"[^\w\s\.-]", "_", base)
    clean = re.sub(r"\s+", "_", clean).strip("._")
    if not clean:
        clean = f"document_{uuid.uuid4().hex[:8]}"
    return clean


def detect_mime_type(header_bytes: bytes) -> str:
    """Inspect first 32 bytes to determine actual file format."""
    if len(header_bytes) >= 3 and header_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(header_bytes) >= 8 and header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(header_bytes) >= 12 and header_bytes.startswith(b"RIFF") and header_bytes[8:12] == b"WEBP":
        return "image/webp"
    if len(header_bytes) >= 4 and header_bytes.startswith(b"%PDF"):
        return "application/pdf"
    if len(header_bytes) >= 4 and (header_bytes.startswith(b"II*\x00") or header_bytes.startswith(b"MM\x00*")):
        return "image/tiff"
    return "application/octet-stream"


def validate_upload(header_bytes: bytes, declared_mime: str, file_size: int, max_size: int) -> Tuple[bool, str, str]:
    """Validate uploaded document bytes, size, and MIME.

    Returns:
        (is_valid, resolved_mime, error_message)
    """
    if file_size <= 0:
        return False, "", "File is empty."

    if file_size > max_size:
        return False, "", f"File size exceeds maximum limit of {max_size} bytes."

    actual_mime = detect_mime_type(header_bytes)

    if actual_mime == "application/octet-stream":
        return False, actual_mime, "Unrecognized or unsupported file format signature."

    # Normalized comparison
    if declared_mime and declared_mime != "application/octet-stream":
        if declared_mime != actual_mime and not (
            declared_mime in ("image/jpg", "image/jpeg") and actual_mime == "image/jpeg"
        ):
            # Log MIME mismatch warning but respect actual binary signature
            pass

    return True, actual_mime, ""
