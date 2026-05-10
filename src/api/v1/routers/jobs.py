"""Job status + SSE event stream.

  GET /api/v1/jobs                    -> list summaries
  GET /api/v1/jobs/{job_id}           -> full manifest + config + error
  GET /api/v1/jobs/{job_id}/stream    -> Server-Sent Events; replays existing
                                        events.jsonl then tails for new ones
                                        until the job reaches a terminal status
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ....data import job_repo
from ....schemas.job import Job, JobSummary
from ....services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobSummary])
def list_jobs() -> list[JobSummary]:
    return [JobSummary(**rec) for rec in job_repo.list_summaries()]


@router.get("/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    return Job(**job_repo.read_manifest(job_id))


@router.get("/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    return StreamingResponse(
        job_service.tail_events(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
