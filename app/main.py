"""
FastAPI application entrypoint with Lambda migration support.
"""

from contextlib import asynccontextmanager
import json
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from alembic import command
from alembic.config import Config

import api.v1 as v1
import api.v2 as v2
from core.config import settings
from core.mira_logger import MiraLogger

# Logger
fastapi_logger = MiraLogger.get_fastapi_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    fastapi_logger.info("Starting Mira Backend API...")
    yield
    fastapi_logger.info("Shutting down Mira Backend API...")


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

# Routers
for router in [
    v1.auth_router,
    v1.demo_router,
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


# Function to run Alembic migrations
def run_migrations():
    fastapi_logger.info("Running Alembic migrations...")
    alembic_cfg = Config("alembic.ini")
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
