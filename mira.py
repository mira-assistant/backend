import uuid
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, Form, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import warnings


from db import get_db_session
from models import Interaction, Person, Conversation


import processors.sentence_processor as SentenceProcessor
from processors.inference_processor import InferenceProcessor
from processors.context_processor import ContextProcessor
from processors.multi_stream_processor import MultiStreamProcessor
from processors.command_processor import CommandProcessor, WakeWordDetector

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    ColorFormatter(fmt="%(levelname)s:\t  %(name)s:\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    from routers.service_router import disable_service

    from routers.service_router import router as service_router
    from routers.interaction_router import router as interaction_router

    app.include_router(service_router)
    app.include_router(interaction_router)

    for interaction in (
        get_db_session().query(Interaction).order_by(Interaction.timestamp.desc()).limit(10).all()
    ):
        status["recent_interactions"].append(interaction.id)

    # Initialize wake words
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
            client_info["connection_runtime"] = round(runtime_seconds, 2)

        # Add score information
        if client_id in scores:
            client_info["score"] = scores[client_id]

    return status




@app.get("/conversations")
def get_recent_conversations(limit: int = 10):
    """Get recent conversations with their interactions."""
    try:
        db = get_db_session()
        try:
            conversations = (
                db.query(Conversation)
                .order_by(Conversation.interactions[0].timestamp.desc())
                .limit(limit)
                .all()
            )

            result = []
            for conv in conversations:
                conv_data = {
                    "id": str(conv.id),
                    "user_ids": [str(user_id) for user_id in conv.user_ids],
                    "interaction_count": len(conv.interactions),
                    "topic_summary": conv.topic_summary,
                }
                result.append(conv_data)

            return result

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")


@app.get("/persons/all")
def get_all_persons():
    """Get all persons in the database."""
    try:
        db = get_db_session()
        persons = db.query(Person).all()
        db.close()

        return persons
    except Exception as e:
        logger.error(f"Error fetching persons: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch persons: {str(e)}")


@app.get("/persons/{person_id}")
def get_person(person_id: str):
    """Get a specific person by ID."""
    try:
        db = get_db_session()
        try:
            person = db.query(Person).filter_by(id=uuid.UUID(person_id)).first()
            if not person:
                raise HTTPException(status_code=404, detail="Person not found")

            return person
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching person: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch person: {str(e)}")


@app.post("/persons/{person_id}/update")
async def update_person(
    person_id: str,
    name: str = Form(...),
    audio: UploadFile = File(...),
    expected_text: str = Form(...),
):
    """Update a person's information, including training their embedding."""
    db = get_db_session()
    try:
        person = db.query(Person).filter_by(id=uuid.UUID(person_id)).first()

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        if name:
            person.name = name  # type: ignore

        if audio and expected_text:
            audio_data = await audio.read()
            if len(audio_data) == 0:
                raise HTTPException(
                    status_code=400, detail="Received empty audio data. Please provide valid audio."
                )

            audio_data = bytearray(audio_data)

            SentenceProcessor.update_voice_embedding(
                person_id=person.id,  # type: ignore
                audio_buffer=audio_data,
                expected_text=expected_text,
            )

        db.commit()
        db.refresh(person)

        return {"message": "Person updated successfully", "person": person}

    except Exception as e:
        logger.error(f"Error updating person: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update person: {str(e)}")

    finally:
        db.close()


@app.get("/streams/best")
def get_best_stream():
    """Get the currently selected best audio stream."""
    best_stream_info = audio_scorer.get_best_stream()

    if not best_stream_info:
        return {"best_stream": None}

    return {"best_stream": best_stream_info}


@app.get("/streams/scores")
def get_all_stream_scores():
    """Get quality scores for all active streams."""

    try:
        clients_info = audio_scorer.clients
        scores = audio_scorer.get_all_stream_scores()
        best_stream = audio_scorer.get_best_stream()

        return {
            "active_streams": len(clients_info),
            "stream_scores": scores,
            "current_best": best_stream,
        }

    except Exception as e:
        logger.error(f"Error getting stream scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream scores: {str(e)}")


@app.get("/streams/{client_id}/info")
def get_client_stream_info(client_id: str):
    """Get detailed stream information about a specific client."""

    if client_id not in audio_scorer.clients:
        raise HTTPException(
            status_code=404, detail=f"Client {client_id} not found in stream scoring"
        )

    client_info = audio_scorer.clients[client_id]
    current_score = audio_scorer.get_all_stream_scores().get(client_id, 0.0)
    best_stream = audio_scorer.get_best_stream()
    is_best_stream = best_stream and best_stream.get("client_id") == client_id

    return {
        "client_id": client_id,
        "quality_metrics": client_info.quality_metrics.__dict__,
        "current_score": round(current_score, 2),
        "is_best_stream": is_best_stream,
        "last_update": client_info.last_update,
    }


@app.get("/service/{client_id}")
def get_client_info(client_id: str):
    """Get detailed information about a specific client."""

    client_dict = status["connected_clients"].get(client_id)

    if not client_dict:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    return client_dict


@app.post("/streams/phone/location")
def update_phone_location(request: dict = Body(...)):
    """Update GPS-based location data for phone tracking."""
    try:
        client_id = request.get("client_id")
        location = request.get("location")

        if not client_id:
            raise HTTPException(status_code=400, detail="client_id is required")
        if not location:
            raise HTTPException(status_code=400, detail="location data is required")

        # Validate location data structure
        required_fields = ["latitude", "longitude"]
        for field in required_fields:
            if field not in location:
                raise HTTPException(status_code=400, detail=f"location.{field} is required")

        # Update phone location in audio scorer
        success = audio_scorer.set_phone_location(client_id, location)

        if not success:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

        logger.info(f"Updated phone location for {client_id}: {location}")

        return {
            "message": f"Phone location updated successfully for {client_id}",
            "location": location,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone location: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update phone location: {str(e)}")


@app.post("/streams/phone/rssi")
def update_phone_rssi(request: dict = Body(...)):
    """Update RSSI-based proximity data for phone tracking to specific client."""
    try:
        phone_client_id = request.get("phone_client_id")  # The phone doing the measurement
        target_client_id = request.get("target_client_id")  # The client being measured
        rssi = request.get("rssi")

        if not phone_client_id:
            raise HTTPException(status_code=400, detail="phone_client_id is required")
        if not target_client_id:
            raise HTTPException(status_code=400, detail="target_client_id is required")
        if rssi is None:
            raise HTTPException(status_code=400, detail="rssi value is required")

        # Update RSSI between phone and target client
        success = audio_scorer.set_phone_rssi(target_client_id, rssi)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Target client {target_client_id} not found"
            )

        logger.info(f"Updated RSSI from {phone_client_id} to {target_client_id}: {rssi} dBm")

        return {
            "message": f"RSSI updated successfully from {phone_client_id} to {target_client_id}",
            "rssi": rssi,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone RSSI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update phone RSSI: {str(e)}")
