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
    return {"message": "Hello Friends Chai Pii Lo"}


@app.post("/movie")
async def movie(data: str):

    results = recommendation(data)
    return {"data": results}



