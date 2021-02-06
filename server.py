from fastapi import FastAPI
from typing import Optional,List
from pydantic import BaseModel
from content import recommendation

# %%
app = FastAPI()

class Content(BaseModel):
    movie: str


@app.get("/")
async def root():
    return {"message": "Go to /docs to open swagger UI, Thank you."}


@app.post("/movie")
async def movie(movie: str):
    print("Recommending on movie -- "+movie)
    results = recommendation(movie)
    return {"data": results}



