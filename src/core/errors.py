from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFound(AppError):
    status_code = 404
    error_code = "not_found"


class ValidationFailure(AppError):
    status_code = 422
    error_code = "validation_error"


class FeasibilityError(AppError):
    status_code = 422
    error_code = "feasibility_error"


class InferenceError(AppError):
    status_code = 500
    error_code = "inference_error"


def _envelope(success: bool, *, code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": success,
        "error": {"code": code, "message": message, "details": details},
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(False, code=exc.error_code, message=exc.message, details=exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(
                False,
                code="request_validation_error",
                message="Request body or params failed validation.",
                details={"errors": exc.errors()},
            ),
        )
