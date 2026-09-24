
from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "model" / "house_price_model.pkl"
)

app = FastAPI(title="House Price Prediction")

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=1)
    bathrooms: int = Field(ge=1)
    age: int = Field(ge=0)
    parking: int = Field(ge=0)
    location: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict")
def predict(data: HouseInput):

    features = pd.DataFrame([
        {
            "area": data.area,
            "bedrooms": data.bedrooms,
            "bathrooms": data.bathrooms,
            "age": data.age,
            "parking": data.parking,
            "location": data.location
        }
    ])

    price = float(model.predict(features)[0])

    price = max(price, 0)

    return {
        "price": round(price, 2),
        "formatted_price": f"₹{price:,.2f}"
    }