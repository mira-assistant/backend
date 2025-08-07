from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import warnings


from db import get_db_session
from models import Interaction

from processors import sentence_processor as SentenceProcessor
from processors.inference_processor import InferenceProcessor
from processors.context_processor import ContextProcessor
from processors.multi_stream_processor import MultiStreamProcessor
from processors.command_processor import CommandProcessor, WakeWordDetector

from interaction_endpoints import router as interaction_router
from person_endpoints import router as person_router
from service_endpoints import router as service_router
from stream_endpoints import router as stream_router
from conversation_endpoints import router as conversation_router

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[92m",
        "ERROR": "\033[91m",
        "WARNING": "\033[93m",
        "DEBUG": "\033[94m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(
    ColorFormatter(fmt="%(levelname)s:\t  %(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

context_processor = ContextProcessor()
audio_scorer = MultiStreamProcessor()
wake_word_detector = WakeWordDetector()
command_processor = CommandProcessor()
inference_processor = InferenceProcessor()

status: dict = {
    "enabled": False,
    "version": "4.3.0",
    "connected_clients": dict(),
    "best_client": None,
    "recent_interactions": deque(maxlen=10),
}

hosting_urls = {
    "localhost": "http://localhost:8000",
    "ankurs-macbook-air": "http://100.75.140.79:8000",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for interaction in (
        get_db_session().query(Interaction).order_by(Interaction.timestamp.desc()).limit(10).all()
    ):
        status["recent_interactions"].append(interaction.id)

    wake_word_detector.add_wake_word("mira cancel", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira exit", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira quit", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira stop", sensitivity=0.5, callback=disable_service)

    app.include_router(interaction_router)
    app.include_router(person_router)
    app.include_router(service_router)
    app.include_router(stream_router)
    app.include_router(conversation_router)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def log_once(message, flag_name=None):
    """Log a message only once during initialization"""
    if flag_name == "advanced" and not globals().get("_advanced_logged", False):
        print(message)
        globals()["_advanced_logged"] = True
    elif flag_name == "context" and not globals().get("_context_logged", False):
        print(message)
        globals()["_context_logged"] = True
    elif flag_name is None and not globals().get("_general_logged", False):
        print(message)
        globals()["_general_logged"] = True


def disable_service():
    status["enabled"] = False
    return {"message": "Service disabled successfully"}


@app.get("/")
def root():
    scores = audio_scorer.get_all_stream_scores()
    status["best_client"] = audio_scorer.get_best_stream()

    current_time = datetime.now(timezone.utc)
    for client_id, client_info in status["connected_clients"].items():
        if "connection_start_time" in client_info:
            connection_start = client_info["connection_start_time"]
            if isinstance(connection_start, str):
                connection_start = datetime.fromisoformat(connection_start.replace("Z", "+00:00"))

            runtime_seconds = (current_time - connection_start).total_seconds()
            client_info["connection_runtime"] = round(runtime_seconds, 2)

        if client_id in scores:
            client_info["score"] = scores[client_id]

    return status
