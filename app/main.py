"""
FastAPI application entrypoint.
"""

from app.core.mira_logger import MiraLogger
from app.core.config import settings

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app.api.v1 as v1
import app.api.v2 as v2


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    MiraLogger.info("Starting Mira Backend API...")
    yield
    MiraLogger.info("Shutting down Mira Backend API...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Mira AI Assistant Backend API",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses."""
    MiraLogger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(v1.conversation_router, prefix="/api/v1")
app.include_router(v1.persons_router, prefix="/api/v1")
app.include_router(v1.streams_router, prefix="/api/v1")
app.include_router(v1.interaction_router, prefix="/api/v1")
app.include_router(v1.service_router, prefix="/api/v1")

app.include_router(v2.conversation_router, prefix="/api/v2")
app.include_router(v2.persons_router, prefix="/api/v2")
app.include_router(v2.streams_router, prefix="/api/v2")
app.include_router(v2.interaction_router, prefix="/api/v2")
app.include_router(v2.service_router, prefix="/api/v2")


@app.get("/")
def root():
    """Root endpoint with system status."""
    return {
        "message": "Mira Backend API",
        "version": settings.app_version,
        "status": "running",
        "stable": "v1",
        "beta": "v2",
    }
