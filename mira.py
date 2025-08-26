from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import warnings


from db import get_db_session
from models import Interaction

from processors.sentence_processor import SentenceProcessor
from processors.inference_processor import InferenceProcessor
from processors.context_processor import ContextProcessor
from processors.multi_stream_processor import MultiStreamProcessor
from processors.command_processor import CommandProcessor, WakeWordDetector

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=".*webrtcvad.*")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[92m",  # Green
        "ERROR": "\033[91m",  # Red
        "WARNING": "\033[93m",  # Yellow
        "DEBUG": "\033[94m",  # Blue
        "CRITICAL": "\033[95m",  # Magenta
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
sentence_processor = SentenceProcessor()

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
    from routers.service_router import router as service_router, disable_service
    from routers.interaction_router import router as interaction_router
    from routers.conversation_router import router as conversation_router
    from routers.persons_router import router as persons_router
    from routers.streams_router import router as streams_router

    app.include_router(service_router)
    app.include_router(interaction_router)
    app.include_router(conversation_router)
    app.include_router(persons_router)
    app.include_router(streams_router)

    for interaction in (
        get_db_session().query(Interaction).order_by(Interaction.timestamp.desc()).limit(10).all()
    ):
        status["recent_interactions"].append(interaction.id)

    wake_word_detector.add_wake_word("mira cancel", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira exit", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira quit", sensitivity=0.5, callback=disable_service)
    wake_word_detector.add_wake_word("mira stop", sensitivity=0.5, callback=disable_service)
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
            client_info["connection_uptime"] = round(runtime_seconds, 2)

        if client_id in scores:
            client_info["score"] = scores[client_id]

    return status
