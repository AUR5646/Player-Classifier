from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os


app = FastAPI()


def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

model = joblib.load(resource_path("decision_tree_model.pkl"))


class PlayerData(BaseModel):
    time: int
    shots: int
    accuracy: float

@app.get("/")
def home():
    return {"message": "Player Classifier API is running"}

@app.post("/predict")
def predict(data: PlayerData):

   
    player = pd.DataFrame(
        [[data.time, data.shots, data.accuracy]],
        columns=["time", "shots", "accuracy"]
    )

    
    prediction = model.predict(player)

    
    return {
        "level": prediction[0]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

    