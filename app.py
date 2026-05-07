from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# โหลดโมเดล
model = joblib.load("Yield_char__CatBoost.joblib")

# ดู feature ของโมเดล
print(model.feature_names_)

class InputData(BaseModel):
    temperature: float
    heating_rate: float
    residence_time: float

@app.post("/predict")
def predict(data: InputData):

    X = pd.DataFrame([{
        "temperature": data.temperature,
        "heating_rate": data.heating_rate,
        "residence_time": data.residence_time
    }])

    pred = model.predict(X)[0]

    return {
        "biochar_yield": float(pred)
    }
