"""Local filesystem implementation of FileStorage."""

import os
from pathlib import Path
import aiofiles
import aiofiles.os

from app.core.config import settings
from app.services.storage.base import FileStorage


class LocalStorageDriver(FileStorage):
    """Stores files on local disk within a secured base directory."""

    def __init__(self, base_dir: str = settings.STORAGE_LOCAL_PATH) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve path and verify it remains strictly within self.base_dir."""
        clean_rel = relative_path.lstrip("/\\")
        target = (self.base_dir / clean_rel).resolve()
        if not str(target).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected: {relative_path}")
        return target

    async def save(self, file_bytes: bytes, destination_path: str) -> str:
        target = self._resolve_safe_path(destination_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "wb") as f:
            await f.write(file_bytes)
        return str(target)

    async def get(self, file_path: str) -> bytes:
        target = self._resolve_safe_path(file_path)
        if not await aiofiles.os.path.exists(target):
            raise FileNotFoundError(f"File not found: {file_path}")
        async with aiofiles.open(target, "rb") as f:
            return await f.read()

    async def delete(self, file_path: str) -> bool:
        target = self._resolve_safe_path(file_path)
        if await aiofiles.os.path.exists(target):
            await aiofiles.os.remove(target)
            return True
        return False

    async def exists(self, file_path: str) -> bool:
        target = self._resolve_safe_path(file_path)
        return await aiofiles.os.path.exists(target)


default_storage = LocalStorageDriver()
