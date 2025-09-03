"""
FastAPI application entrypoint.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.db.init_db import init_db
from app.api.v1 import auth, assistant, tasks

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global status for backward compatibility
status: dict = {
    "enabled": False,
    "version": settings.app_version,
    "connected_clients": dict(),
    "best_client": None,
    "recent_interactions": deque(maxlen=10),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Mira Backend API...")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Load recent interactions
    from app.db.session import get_db_session
    from app.models.interaction import Interaction

    db = get_db_session()
    try:
        for interaction in (
            db.query(Interaction).order_by(Interaction.timestamp.desc()).limit(10).all()
        ):
            status["recent_interactions"].append(interaction.id)
    finally:
        db.close()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Mira Backend API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Mira AI Assistant Backend API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(assistant.router, prefix="/api/v1/assistant", tags=["assistant"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])


@app.get("/")
def root():
    """Root endpoint with system status."""
    return {
        "message": "Mira Backend API",
        "version": settings.app_version,
        "status": "running",
        "status_info": status
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.app_version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

