"""Format-aware file-boundary checks and safe trailing-data carving."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def _base_result(raw: bytes, format_name: str, boundary: Optional[int], reason: str) -> dict:
    trailing = raw[boundary:] if boundary is not None and boundary < len(raw) else b""
    return {
        "format": format_name,
        "boundary_offset": boundary,
        "file_size": len(raw),
        "trailing_bytes": len(trailing),
        "has_trailing_data": bool(trailing),
        "reason": reason,
        "trailing_data": trailing,
        "confidence": "HIGH" if boundary is not None else "LIMITED",
    }


def _png_boundary(raw: bytes) -> tuple[Optional[int], str]:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, "PNG signature is invalid"
    pos = 8
    while pos + 12 <= len(raw):
        length = int.from_bytes(raw[pos:pos + 4], "big")
        end = pos + 12 + length
        if end > len(raw):
            return None, "PNG chunk extends past the available file bytes"
        chunk_type = raw[pos + 4:pos + 8]
        if chunk_type == b"IEND":
            return end, "PNG IEND chunk found"
        pos = end
    return None, "PNG IEND chunk was not found"


def audit_file_structure(raw: bytes, filename: str = "") -> Dict[str, object]:
    """Find bytes after the format's structural end marker, when defined.

    The carved bytes are returned in ``trailing_data`` for a researcher to
    download. They are never executed or parsed as code by this module.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("raw must be bytes")
    raw = bytes(raw)
    suffix = Path(filename).suffix.lower()

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        boundary, reason = _png_boundary(raw)
        result = _base_result(raw, "PNG", boundary, reason)
        expected = {".png", ""}
    elif raw.startswith(b"\xff\xd8\xff"):
        eoi = raw.rfind(b"\xff\xd9")
        boundary = eoi + 2 if eoi >= 0 else None
        result = _base_result(raw, "JPEG", boundary,
                              "JPEG EOI marker found" if boundary else "JPEG EOI marker was not found")
        expected = {".jpg", ".jpeg", ""}
    elif raw.startswith((b"GIF87a", b"GIF89a")):
        trailer = raw.rfind(b"\x3b")
        boundary = trailer + 1 if trailer >= 0 else None
        result = _base_result(raw, "GIF", boundary,
                              "GIF trailer found" if boundary else "GIF trailer was not found")
        expected = {".gif", ""}
    elif raw.startswith(b"BM") and len(raw) >= 14:
        declared = int.from_bytes(raw[2:6], "little")
        boundary = declared if 14 <= declared <= len(raw) else None
        result = _base_result(raw, "BMP", boundary,
                              "BMP declared file size used" if boundary else "BMP declared file size is unavailable or inconsistent")
        expected = {".bmp", ""}
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP" and len(raw) >= 12:
        declared = int.from_bytes(raw[4:8], "little") + 8
        boundary = declared if 12 <= declared <= len(raw) else None
        result = _base_result(raw, "WEBP", boundary,
                              "RIFF container size used" if boundary else "RIFF container size is inconsistent")
        expected = {".webp", ""}
    else:
        result = _base_result(raw, "Unknown", None,
                              "No reliable end marker is implemented for this format")
        expected = {suffix, ""}

    result["extension_matches_format"] = suffix in expected
    result["filename"] = filename
    return result
