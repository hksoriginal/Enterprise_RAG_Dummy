import asyncio
import logging
import os
import sys

# Add root project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI

from UI.TelegramBot import telegram_bot
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from API.Routes.get_answer_route import get_answer_router
from API.Routes.ingest_route import ingest_router
from API.Routes.retrieval_route import retrieval_router


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------------------------

APP_NAME = os.getenv(
    "APP_NAME",
    "production-rag-api",
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

HOST = os.getenv(
    "HOST",
    "0.0.0.0",
)

PORT = int(
    os.getenv(
        "PORT",
        8100,
    )
)

DEBUG = os.getenv(
    "DEBUG",
    "false",
).lower() == "true"


# ------------------------------------------------------------------------------
# Application Lifespan
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator:
    """
    Startup and shutdown lifecycle manager.
    """

    try:
        logger.info(
            "Starting application | name=%s | version=%s",
            APP_NAME,
            APP_VERSION,
        )

        # ----------------------------------------------------------------------
        # Startup Logic
        # ----------------------------------------------------------------------

        app.state.telegram_bot_task = await telegram_bot.start_telegram_bot()
        if app.state.telegram_bot_task is None:
            logger.warning("Telegram bot was not started")

        logger.info("Application startup completed")

        yield

    except Exception:
        logger.exception(
            "Application startup failed"
        )
        raise

    finally:
        # ----------------------------------------------------------------------
        # Shutdown Logic
        # ----------------------------------------------------------------------

        if getattr(app.state, "telegram_bot_task", None) is not None:
            await telegram_bot.stop_telegram_bot(app.state.telegram_bot_task)

        logger.info("Shutting down application")


# ------------------------------------------------------------------------------
# FastAPI App Initialization
# ------------------------------------------------------------------------------

app = FastAPI(
    title="Production RAG API",
    description=(
        "Production-grade Retrieval-Augmented "
        "Generation API"
    ),
    version=APP_VERSION,
    debug=DEBUG,
    lifespan=lifespan,
)


# ------------------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Replace in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Global Exception Handler
# ------------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc,
):
    """
    Catch unhandled exceptions globally.
    """

    logger.exception(
        "Unhandled application exception | path=%s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": (
                "Internal server error"
            ),
        },
    )


# ------------------------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------------------------

@app.get("/")
async def root():
    """
    Root endpoint.
    """

    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
    }


# ------------------------------------------------------------------------------
# Router Registration
# ------------------------------------------------------------------------------

app.include_router(ingest_router)
app.include_router(retrieval_router)
app.include_router(get_answer_router)


# ------------------------------------------------------------------------------
# Uvicorn Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    logger.info(
        "Starting Uvicorn server | host=%s | port=%d",
        HOST,
        PORT,
    )

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info",
    )
