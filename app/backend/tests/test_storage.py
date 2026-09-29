"""Tests for LocalStorageDriver."""

import pytest
from app.services.storage.local import LocalStorageDriver


@pytest.mark.asyncio
async def test_local_storage_lifecycle(tmp_path):
    driver = LocalStorageDriver(base_dir=str(tmp_path))

    data = b"sample test file content"
    dest = "2026/09/doc123/file.jpg"

    # Save
    saved_path = await driver.save(data, dest)
    assert await driver.exists(saved_path)

    # Get
    content = await driver.get(saved_path)
    assert content == data

    # Delete
    deleted = await driver.delete(saved_path)
    assert deleted is True
    assert not await driver.exists(saved_path)


@pytest.mark.asyncio
async def test_local_storage_traversal_protection(tmp_path):
    driver = LocalStorageDriver(base_dir=str(tmp_path))

    with pytest.raises(ValueError, match="Path traversal detected"):
        await driver.save(b"malicious", "../../etc/passwd")
