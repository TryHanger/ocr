"""Background pipeline worker process."""

import asyncio
import os
import signal
import sys
import time
from sqlalchemy import desc, select

from app.core.config import settings
from app.core.enums import JobStatus
from app.core.logging import logger
from app.db.session import async_session_factory
from app.models.job import ProcessingJob
from app.services.pipeline_service import PipelineService


class PipelineWorker:
    """Polls for pending processing jobs and executes pipeline stages."""

    def __init__(self, poll_interval: float = settings.WORKER_POLL_INTERVAL_SECONDS) -> None:
        self.poll_interval = poll_interval
        self._running = False

    async def _fetch_next_job_id(self) -> str | None:
        """Find the next pending or retrying job ID."""
        async with async_session_factory() as session:
            stmt = (
                select(ProcessingJob.id)
                .where(
                    ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.RETRYING.value])
                )
                .order_by(desc(ProcessingJob.priority), ProcessingJob.created_at.asc())
                .limit(1)
            )
            res = await session.execute(stmt)
            return res.scalar_one_or_none()

    async def run(self) -> None:
        """Run worker loop."""
        self._running = True
        logger.info(
            "Document AI Pipeline Worker started",
            extra={"poll_interval": self.poll_interval},
        )

        while self._running:
            try:
                job_id = await self._fetch_next_job_id()
                if job_id:
                    async with async_session_factory() as session:
                        service = PipelineService(session)
                        await service.execute_job(job_id)
                else:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker encountered unexpected loop error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        logger.info("Document AI Pipeline Worker stopped cleanly")

    def stop(self) -> None:
        self._running = False


async def main() -> None:
    worker = PipelineWorker()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except NotImplementedError:
            # Signal handling on Windows
            pass

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
