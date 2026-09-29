"""Tests for upload validation and security utilities."""

from app.core.security import detect_mime_type, sanitize_filename, validate_upload


def test_sanitize_filename():
    assert sanitize_filename("../../../malicious file.jpg") == "malicious_file.jpg"
    assert sanitize_filename("receipt#1!2@3.png") == "receipt_1_2_3.png"
    assert sanitize_filename("") != ""


def test_detect_mime_type():
    assert detect_mime_type(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "image/jpeg"
    assert detect_mime_type(b"\x89PNG\r\n\x1a\n\x00\x00") == "image/png"
    assert detect_mime_type(b"%PDF-1.4\n") == "application/pdf"
    assert detect_mime_type(b"random arbitrary bytes") == "application/octet-stream"


def test_validate_upload():
    valid_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    is_valid, mime, err = validate_upload(valid_jpeg[:32], "image/jpeg", len(valid_jpeg), 1000)
    assert is_valid is True
    assert mime == "image/jpeg"
    assert err == ""

    # Empty
    is_valid, _, err = validate_upload(b"", "image/jpeg", 0, 1000)
    assert is_valid is False
    assert "empty" in err.lower()

    # Oversized
    is_valid, _, err = validate_upload(valid_jpeg[:32], "image/jpeg", 2000, 1000)
    assert is_valid is False
    assert "limit" in err.lower()

    # Fake MIME
    fake_file = b"This is a text file pretending to be image"
    is_valid, _, err = validate_upload(fake_file[:32], "image/jpeg", len(fake_file), 1000)
    assert is_valid is False
