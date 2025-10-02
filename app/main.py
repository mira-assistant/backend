"""
FastAPI application entrypoint with Lambda migration support.
"""

import signal
from contextlib import asynccontextmanager

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from starlette.middleware.sessions import SessionMiddleware

import app.api.v1 as v1
import app.api.v2 as v2
from app.core.config import settings
from app.core.mira_logger import MiraLogger

# Logger
fastapi_logger = MiraLogger.get_fastapi_logger()

# Track if AWS secrets have been loaded
_aws_secrets_loaded = False


def cleanup_resources():
    """Force cleanup of all resources."""
    try:
        from app.services.service_registry import service_registry

        service_registry.cleanup_all()
    except Exception as e:
        fastapi_logger.error(f"Error during cleanup: {e}")

    # Force garbage collection
    import gc

    gc.collect()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    import os

    fastapi_logger.info(f"Received signal {signum}, forcing shutdown...")
    cleanup_resources()

    # Use os._exit to bypass Python cleanup and force immediate exit
    # This is necessary because ML libraries create non-daemon C threads
    os._exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _aws_secrets_loaded

    fastapi_logger.info("Starting Mira Backend API...")

    # Load AWS secrets if running in Lambda environment
    import os

    if os.environ.get("AWS_EXECUTION_ENV") and not _aws_secrets_loaded:
        fastapi_logger.info("Detected AWS Lambda environment, loading secrets...")
        settings.load_aws_secrets("mira-secrets")
        fastapi_logger.info(f"Database URL: {settings.database_url}")
        _aws_secrets_loaded = True

    yield

    fastapi_logger.info("Shutting down Mira Backend API...")
    cleanup_resources()
    fastapi_logger.info("Shutdown complete")


# Initialize FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Mira AI Assistant Backend API",
    lifespan=lifespan,
)

asgi_handler = Mangum(app, api_gateway_base_path="/")


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    fastapi_logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Session middleware for OAuth2 flows
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=3600,  # 1 hour
    same_site="lax",
    https_only=False,  # Set to True in production with HTTPS
)

# Routers
for router in [
    v1.conversation_router,
    v1.persons_router,
    v1.streams_router,
    v1.interaction_router,
    v1.service_router,
]:
    app.include_router(router, prefix="/api/v1")

for router in [
    v2.conversation_router,
    v2.persons_router,
    v2.streams_router,
    v2.interaction_router,
    v2.service_router,
    v2.auth_router,
]:
    app.include_router(router, prefix="/api/v2")


@app.get("/")
def root():
    return {
        "message": "Mira Backend API",
        "version": settings.app_version,
        "status": "running",
        "stable": "v1",
        "beta": "v2",
    }


def run_migrations():
    # Load secrets explicitly
    settings.load_aws_secrets("mira-secrets")

    alembic_cfg = Config("alembic.ini")
    fastapi_logger.info("Running Alembic migrations...")
    try:
        command.upgrade(alembic_cfg, "head")
        fastapi_logger.info("Migrations completed successfully.")
        return {"status": "success", "message": "Database migrated successfully."}
    except Exception as e:
        fastapi_logger.exception("Migration failed")
        return {"status": "error", "message": str(e)}


# Lambda handler
def handler(event, context):
    """
    Mangum Lambda entrypoint.
    If the payload contains {"action": "run_migrations"}, it runs Alembic migrations.
    Otherwise, it serves the FastAPI API.
    """
    if isinstance(event, dict) and event.get("action") == "run_migrations":
        return run_migrations()

    # Otherwise, route through Mangum
    return asgi_handler(event, context)
