from mira import (
    logger,
)
from db import get_db_session
import processors.sentence_processor as SentenceProcessor
from models import Person
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import uuid

router = APIRouter(prefix="/persons")


@router.get("/all")
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


@router.get("/{person_id}")
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


@router.post("/{person_id}/update")
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
