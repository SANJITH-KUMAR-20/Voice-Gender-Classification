from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import *
from predict import *

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://localhost:3000",
    "null"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/get_result")
async def get_result(file : UploadFile = File()):
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    # gender = predict("", file)

    return Response(content=gender,media_type="application/json")
