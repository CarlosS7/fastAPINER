import spacy

from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
from typing import List

model_dir = Path('model')
nlp = spacy.load(model_dir)

api = FastAPI()

class Input(BaseModel):
    sentence: str

class Extraction(BaseModel):
    name: str
    content: str

class Output(BaseModel):
    extractions: List[Extraction]

@api.post("/extractions", response_model=Output)
def extractions(input: Input):
    document = nlp(input.sentence)

    extractions = []
    for entity in document.ents:
        extraction = {}
        extraction["name"] = entity.label_
        extraction["content"] = entity.text
        extractions.append(extraction)

    return {"extractions": extractions}