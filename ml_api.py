from fastapi import FastAPI 
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
    area : float
    bedrooms : float
    bathrooms : float
    stories : float
    mainroad : int
    guestroom : int
    basement : int
    hotwaterheating : int
    airconditioning : float
    parking : float 
    prefarea : int 
    furnishingstatus_furnished : int
    furnishingstatus_semi_furnished : int
    furnishingstatus_unfurnished : int

with open("housing_price_prediction.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.post("/")

async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    y = model.predict(df)
    return {"prediction" : float(y)} 