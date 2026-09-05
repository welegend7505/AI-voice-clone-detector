from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(title="Voice Clone Detection API - SIH26104")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a hackathon demo; tighten for anything real
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Run with: uvicorn main:app --reload --port 8000
