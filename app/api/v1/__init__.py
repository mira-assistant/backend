from api.v1.auth_router import router as auth_router
from api.v1.conversation_router import router as conversation_router
from api.v1.demo_router import router as demo_router
from api.v1.interaction_router import router as interaction_router
from api.v1.persons_router import router as persons_router
from api.v1.service_router import router as service_router
from api.v1.streams_router import router as streams_router

__all__ = [
    "auth_router",
    "demo_router",
    "conversation_router",
    "streams_router",
    "persons_router",
    "interaction_router",
    "service_router",
]
