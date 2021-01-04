from fastapi import FastAPI
from typing import Optional,List
from pydantic import BaseModel

# %%
app = FastAPI()

class Content(BaseModel):
    movie: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/movie")
async def movie(data:Content):
    
    return {"data":data}


# class Item(BaseModel):
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None


# app = FastAPI()


# @app.post("/items/")
# async def create_item(item: Item):
#     return item