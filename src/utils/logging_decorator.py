import json
import time
import functools
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from ..logger import logger


def log_endpoint(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()

        # ===== EXTRACT REQUEST DATA =====
        request_data = {}

        for key, value in kwargs.items():
            if isinstance(value, BaseModel):
                request_data[key] = jsonable_encoder(value)
            else:
                request_data[key] = value

        try:
            logger.info(
                "\n=== REQUEST ===\n"
                f"{json.dumps(request_data, indent=2)}"
            )
        except Exception as e:
            logger.error(f"Failed to log request: {e}")

        # ===== CALL ENDPOINT =====
        response = await func(*args, **kwargs)

        # ===== SERIALIZE RESPONSE =====
        response_data = jsonable_encoder(response)

        duration = round((time.time() - start) * 1000, 2)

        try:
            logger.info(
                "=== RESPONSE ===\n"
                f"{json.dumps(response_data, indent=2)}\n"
                f"=== {duration} ms ===\n"
            )
        except Exception as e:
            logger.error(f"Failed to log response: {e}")

        return response

    return wrapper