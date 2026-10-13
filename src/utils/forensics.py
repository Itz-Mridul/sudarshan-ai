"""Backward-compatible wrappers for the forensic screening modules."""

from forensics.rs_steganalysis import analyse_rs
from forensics.structural_audit import audit_file_structure


def check_eof_padding(data):
    """Preserve the original API while using the format-aware auditor."""
    result = audit_file_structure(data)
    return {
        "has_padding": result["has_trailing_data"],
        "padding_size": result["trailing_bytes"],
        "padding_data": result["trailing_data"][:10240],
        "message": (
            f"Found {result['trailing_bytes']} bytes after the {result['format']} boundary."
            if result["has_trailing_data"] else result["reason"]
        ),
        "format": result["format"],
        "boundary_offset": result["boundary_offset"],
    }


def fast_rs_analysis(image_np):
    """Return the RS screening score for callers of the legacy helper."""
    return float(analyse_rs(image_np)["screening_score"])
