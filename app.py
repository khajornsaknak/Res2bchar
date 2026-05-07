from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# โหลดโมเดล
model = joblib.load("Yield_char__CatBoost.joblib")

@app.get("/")
def home():
    return {"message": "Biochar API Running"}

@app.get("/predict")
def predict(
    temperature: float,
    time: float,
    ash: float,
    vm: float,
    fc: float
):

    # สร้าง DataFrame
    X = pd.DataFrame([{
        "temperature": temperature,
        "time": time,
        "ash": ash,
        "vm": vm,
        "fc": fc
    }])

    # ทำนาย
    pred = model.predict(X)[0]

    return {
        "predicted_yield": float(pred)
    }
