"""
Shared utilities for API routers.
"""

from pathlib import Path

from fastapi import HTTPException


def validate_filename(filename: str, allowed_extension: str | None = None) -> str:
    """
    Validate and sanitize an uploaded filename.
    
    Prevents path traversal attacks and optionally enforces file extension.
    
    Args:
        filename: Raw filename from upload
        allowed_extension: Required extension (e.g., '.pdf', '.csv') or None for any
        
    Returns:
        Sanitized filename (basename only)
        
    Raises:
        HTTPException: If filename is invalid or has wrong extension
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Extract basename to prevent path traversal
    clean_name = Path(filename).name
    
    # Check for path traversal attempts
    if "/" in clean_name or "\\" in clean_name or ".." in clean_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Check extension if required
    if allowed_extension and not clean_name.lower().endswith(allowed_extension.lower()):
        raise HTTPException(
            status_code=400, 
            detail=f"Only {allowed_extension} files are supported"
        )
    
    return clean_name


def validate_path_within_directory(file_path: Path, directory: Path) -> None:
    """
    Verify that a file path is within the expected directory.
    
    Args:
        file_path: Path to validate
        directory: Directory the path should be within
        
    Raises:
        HTTPException: If path escapes the directory
    """
    try:
        file_path.resolve().relative_to(directory.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")