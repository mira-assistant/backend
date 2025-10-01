import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, UploadFile
from sqlalchemy.orm import Session

import app.db as db
import app.models as models
from app.core.mira_logger import MiraLogger
from app.services.service_factory import get_sentence_processor

router = APIRouter(prefix="/persons")


@router.get("/{person_id}")
def get_person(
    person_id: str = Path(..., description="The ID of the person"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get a specific person by ID within a specific network."""

    try:
        network_uuid = uuid.UUID(network_id)
        person_uuid = uuid.UUID(person_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    person = (
        db.query(models.Person)
        .filter(models.Person.network_id == network_uuid)
        .filter(models.Person.id == person_uuid)
        .first()
    )

    if not person:
        raise HTTPException(status_code=404, detail="Person not found in this network")

    return person


@router.post("/{person_id}/update")
async def update_person(
    person_id: str = Path(..., description="The ID of the person to update"),
    network_id: str = Path(..., description="The ID of the network"),
    name: str = Form(None, description="The new name for the person"),
    audio: UploadFile = File(
        None, description="Audio file for voice embedding training"
    ),
    expected_text: str = Form(None, description="Expected text spoken in the audio"),
    db: Session = Depends(db.get_db),
):
    """Update a person's information, including training their voice embedding."""

    try:
        # Validate UUIDs
        try:
            network_uuid = uuid.UUID(network_id)
            person_uuid = uuid.UUID(person_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid UUID format")

        # Find the person in the network
        person = (
            db.query(models.Person)
            .filter(
                models.Person.id == person_uuid,
                models.Person.network_id == network_uuid,
            )
            .first()
        )

        if not person:
            raise HTTPException(
                status_code=404, detail="Person not found in this network"
            )

        # Update name if provided
        if name:
            person.name = name  # type: ignore
            MiraLogger.info(f"Updated name for person {person_id} to: {name}")

        # Update voice embedding if audio and expected text are provided
        if audio and expected_text:
            audio_data = await audio.read()
            if len(audio_data) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Received empty audio data. Please provide valid audio.",
                )

            MiraLogger.info(
                f"Training voice embedding for person {person_id} with {len(audio_data)} bytes of audio"
            )

            # Get the sentence processor for this network
            sentence_processor = get_sentence_processor(network_id)

            # Convert audio to bytearray for processing
            audio_bytearray = bytearray(audio_data)

            # Update the voice embedding using the sentence processor
            sentence_processor.update_voice_embedding(
                person_id=person_id,
                audio_buffer=audio_bytearray,
                expected_text=expected_text,
            )

            MiraLogger.info(
                f"Successfully updated voice embedding for person {person_id} with expected text: '{expected_text}'"
            )

        # Commit changes to database
        db.commit()
        db.refresh(person)

        return {
            "message": "Person updated successfully",
            "person": {
                "id": str(person.id),
                "name": person.name,
                "index": person.index,
                "network_id": person.network_id,
                "has_voice_embedding": person.voice_embedding is not None,
                "cluster_id": person.cluster_id,
            },
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        MiraLogger.error(f"Error updating person {person_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to update person: {str(e)}"
        )


@router.get("/")
def get_all_persons(
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get all persons in a specific network."""

    try:
        network_uuid = uuid.UUID(network_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format")

    persons = (
        db.query(models.Person).filter(models.Person.network_id == network_uuid).all()
    )

    return {
        "network_id": network_id,
        "persons": [
            {
                "id": str(person.id),
                "name": person.name,
                "index": person.index,
                "has_voice_embedding": person.voice_embedding is not None,
                "cluster_id": person.cluster_id,
                "created_at": person.created_at.isoformat()
                if person.created_at  # pyright: ignore[reportGeneralTypeIssues]
                else None,  # type: ignore
            }
            for person in persons
        ],
        "total_count": len(persons),
    }


@router.delete("/{person_id}")
def delete_person(
    person_id: str = Path(..., description="The ID of the person to delete"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Delete a person from the network."""

    try:
        # Validate UUIDs
        try:
            network_uuid = uuid.UUID(network_id)
            person_uuid = uuid.UUID(person_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid UUID format")

        # Find the person in the network
        person = (
            db.query(models.Person)
            .filter(
                models.Person.id == person_uuid,
                models.Person.network_id == network_uuid,
            )
            .first()
        )

        if not person:
            raise HTTPException(
                status_code=404, detail="Person not found in this network"
            )

        # Delete the person
        db.delete(person)
        db.commit()

        MiraLogger.info(f"Deleted person {person_id} from network {network_id}")

        return {
            "message": f"Person {person_id} deleted successfully",
            "deleted_person_id": person_id,
            "network_id": network_id,
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        MiraLogger.error(f"Error deleting person {person_id}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete person: {str(e)}"
        )
