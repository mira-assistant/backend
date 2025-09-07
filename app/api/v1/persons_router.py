from sqlalchemy.orm import Session
import db
import models
from fastapi import APIRouter, Depends, Path, UploadFile, File, Form, HTTPException
from core.mira_logger import MiraLogger

router = APIRouter(prefix="/{network_id}/persons")


@router.get("/{person_id}")
def get_person(
    person_id: str = Path(..., description="The ID of the person"),
    network_id: str = Path(..., description="The ID of the network"),
    db: Session = Depends(db.get_db),
):
    """Get a specific person by ID within a specific network."""

    person = (
        db.query(models.Person)
        .filter(models.Person.network_id == network_id)
        .filter(models.Person.id == person_id)
        .first()
    )

    if not person:
        raise HTTPException(status_code=404, detail="Person not found in this network")

    return person


# @router.post("/{person_id}/update")
# async def update_person(
#     person_id: str,
#     network_id: str = Path(..., description="The ID of the network"),
#     name: str = Form(...),
#     audio: UploadFile = File(...),
#     expected_text: str = Form(...),
#     db: Session = Depends(db.get_db),
# ):
#     """Update a person's information, including training their embedding."""
#     try:
#         person = db.query(models.Person).filter(
#             models.Person.id == person_id,
#             models.Person.network_id == network_id
#         ).first()

#         if not person:
#             raise HTTPException(status_code=404, detail="Person not found in this network")

#         if name:
#             person.name = name  # type: ignore

#         if audio and expected_text:
#             audio_data = await audio.read()
#             if len(audio_data) == 0:
#                 raise HTTPException(
#                     status_code=400, detail="Received empty audio data. Please provide valid audio."
#                 )

#             # TODO: Implement voice embedding update
#             # This would require importing and using the sentence_processor
#             MiraLogger.info(f"Audio data received for person {person.id}: {len(audio_data)} bytes")

#         db.commit()
#         db.refresh(person)

#         return {"message": "Person updated successfully", "person": person}

#     except Exception as e:
#         MiraLogger.error(f"Error updating person: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to update person: {str(e)}")
