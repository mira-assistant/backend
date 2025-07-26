from fastapi import FastAPI, HTTPException
from run_inference import send_prompt
from models import Interaction
from db import get_db_session
import uvicorn
from sentence_processor import transcribe_interaction


if __name__ == "__main__":
    uvicorn.run("mira:app", host="0.0.0.0", port=8000, reload=True)

app = FastAPI()


status: dict = {
    "version": "1.0.0",
    "listening_clients": list(),
    "enabled": False,
}


@app.get("/")
def root():
    return status


@app.post("/register_client")
def register_client(client_id: str):
    status["listening_clients"].append(client_id)
    print("Client registered:", client_id)
    return status


@app.post("/deregister_client")
def deregister_client(client_id: str):
    if client_id in status["listening_clients"]:
        status["listening_clients"].remove(client_id)
    else:
        raise HTTPException(status_code=404, detail="Client not found")

    print("Client deregistered:", client_id)

    return status


@app.patch("/enable")
def enable_service():
    status["enabled"] = True
    print("Mira enabled")
    return status


@app.patch("/disable")
def disable_service():
    status["enabled"] = False
    print("Mira disabled")
    return status


@app.post("/process_interaction")
def process_interaction(sentence_buf: bytearray):
    interaction = Interaction(**transcribe_interaction(sentence_buf))

    db = get_db_session()
    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    return interaction


@app.post("/inference")
def inference_endpoint(interaction_id):
    interaction = (
        get_db_session().query(Interaction).filter_by(id=interaction_id).first()
    )

    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")

    if "mira" in interaction.text.lower() and any(
        cancelCMD in interaction.text.lower() for cancelCMD in ("cancel", "exit")
    ):
        disable_service()
        return status

    # context_processor = create_context_processor()
    # context, has_intent = context_processor.process_input(str(interaction.text))

    # if not has_intent:
    #     return {"message": "Intent not recognized, no inference performed."}

    response = send_prompt(prompt=str(interaction.text))
    return response
