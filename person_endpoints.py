import uuid
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from db import get_db_session_context, handle_db_error
from models import Person
from processors import sentence_processor as SentenceProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("/all")
@handle_db_error("get_all_persons")
def get_all_persons():
    """Get all persons in the database."""
    with get_db_session_context() as db:
        persons = db.query(Person).all()
        return persons


@router.get("/{person_id}")
@handle_db_error("get_person")
def get_person(person_id: str):
    """Get a specific person by ID."""
    with get_db_session_context() as db:
        person = db.query(Person).filter_by(id=uuid.UUID(person_id)).first()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        return person


@router.post("/{person_id}/update")
@handle_db_error("update_person")
async def update_person(
    person_id: str,
    name: str = Form(...),
    audio: UploadFile = File(...),
    expected_text: str = Form(...),
):
    """Update a person's information, including training their embedding."""
    with get_db_session_context() as db:
        person = db.query(Person).filter_by(id=uuid.UUID(person_id)).first()

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        if name:
            person.name = name

        if audio and expected_text:
            audio_data = await audio.read()
            if len(audio_data) == 0:
                raise HTTPException(
                    status_code=400, detail="Received empty audio data. Please provide valid audio."
                )

            audio_data = bytearray(audio_data)

            SentenceProcessor.update_voice_embedding(
                person_id=person.id,
                audio_buffer=audio_data,
                expected_text=expected_text,
            )

        db.flush()

        return {"message": "Person updated successfully", "person": person}