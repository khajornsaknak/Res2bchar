from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = "Yield_char__CatBoost.joblib"

model = joblib.load(BASE_DIR / MODEL_FILE)

feature_order = list(model.feature_names_in_)

# =====================================================
# APP
# =====================================================

app = FastAPI()

# =====================================================
# INPUT MODEL
# =====================================================

class InputData(BaseModel):

    Temp: float
    RT: float
    HR: float

    VM_bio: float
    Ash_bio: float
    FC_bio: float

    H_bio: float
    C_bio: float
    O_bio: float


# =====================================================
# SAFE DIV
# =====================================================

def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():
    return {"message": "Biochar API is running"}


# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
def predict(data: InputData):

    vm = data.VM_bio
    ash = data.Ash_bio
    fc = data.FC_bio
    h = data.H_bio
    c = data.C_bio
    o = data.O_bio

    row_map = {

        "Temp": data.Temp,
        "RT": data.RT,
        "HR": data.HR,

        "VM_bio": vm,
        "Ash_bio": ash,
        "FC_bio": fc,

        "H/C_bio": safe_div(h, c),
        "O/C_bio": safe_div(o, c),

        "VM_to_Ash": safe_div(vm, ash),
        "FC_to_Ash": safe_div(fc, ash),

        "Temp_x_RT": data.Temp * data.RT,
        "Temp_x_HR": data.Temp * data.HR,
        "Temp_x_Ash": data.Temp * ash,
        "Temp_x_VM": data.Temp * vm,

        "Feedstock_type": "Unknown",
        "FC_bio_derived": fc,
        "O_bio_derived": o
    }

    X = pd.DataFrame(
        [[row_map[col] for col in feature_order]],
        columns=feature_order
    )

    pred = float(model.predict(X)[0])

    return {
        "predicted_yield": round(pred, 4)
    }
