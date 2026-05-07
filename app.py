from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# โหลดโมเดล
model = joblib.load("Yield_char__CatBoost.joblib")

# รูปแบบข้อมูลรับเข้า
class InputData(BaseModel):
    temperature: float
    heating_rate: float
    residence_time: float

# API predict
@app.post("/predict")
def predict(data: InputData):

    # แปลงเป็น DataFrame
    X = pd.DataFrame([{
        "temperature": data.temperature,
        "heating_rate": data.heating_rate,
        "residence_time": data.residence_time
    }])

    # ทำนาย
    pred = model.predict(X)[0]

    return {
        "biochar_yield": float(pred)
    }