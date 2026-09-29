"""Storage abstraction protocol."""

from typing import BinaryIO, Protocol, runtime_checkable


@runtime_checkable
class FileStorage(Protocol):
    """File storage contract supporting local filesystem, S3, or MinIO."""

    async def save(self, file_bytes: bytes, destination_path: str) -> str:
        """Persist file bytes to storage and return canonical stored path or URI."""
        ...

    async def get(self, file_path: str) -> bytes:
        """Retrieve binary file contents from storage."""
        ...

    async def delete(self, file_path: str) -> bool:
        """Remove file from storage."""
        ...

    async def exists(self, file_path: str) -> bool:
        """Check if file exists in storage."""
        ...
