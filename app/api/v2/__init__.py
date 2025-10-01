from api.v2.auth_router import router as auth_router
from api.v2.conversation_router import router as conversation_router
from api.v2.interaction_router import router as interaction_router
from api.v2.persons_router import router as persons_router
from api.v2.service_router import router as service_router
from api.v2.streams_router import router as streams_router

__all__ = [
    "auth_router",
    "conversation_router",
    "streams_router",
    "persons_router",
    "interaction_router",
    "service_router",
]
