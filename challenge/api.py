from datetime import datetime, timezone

import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, validator

from challenge.model import DelayModel

VALID_OPERA_VALUES = [
    "american airlines",
    "air canada",
    "air france",
    "aeromexico",
    "aerolineas argentinas",
    "austral",
    "avianca",
    "alitalia",
    "british airways",
    "copa air",
    "delta air",
    "gol trans",
    "iberia",
    "k.l.m.",
    "qantas airways",
    "united airlines",
    "grupo latam",
    "sky airline",
    "latin american wings",
    "plus ultra lineas aereas",
    "jetsmart spa",
    "oceanair linhas aereas",
    "lacsa",
]

VALID_TIPO_VUELO_VALUES = [
    "I",
    "N",
]

VALID_MES_VALUES = range(1, 13)


app = fastapi.FastAPI()
model = DelayModel()
model.load_model("models")


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def valid_opera(cls, opera_value: str):
        if opera_value.lower() not in VALID_OPERA_VALUES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Value for tipo vuelo not valid. Recieved {opera_value}, "
                    f"expected one from {VALID_OPERA_VALUES}"
                ),
            )
        return opera_value

    @validator("TIPOVUELO")
    def valid_tipo_vuelo(cls, tipo_vuelo_value: str):
        if tipo_vuelo_value.capitalize() not in VALID_TIPO_VUELO_VALUES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Value for tipo vuelo not valid. Recieved {tipo_vuelo_value}, "
                    f"expected one from {VALID_TIPO_VUELO_VALUES}"
                ),
            )
        return tipo_vuelo_value

    @validator("MES")
    def valid_mes(cls, mes_value: int):
        if mes_value not in VALID_MES_VALUES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Value for tipo vuelo not valid. Recieved {mes_value}, "
                    f"expected one from {VALID_MES_VALUES}"
                ),
            )
        return mes_value


class FlightData(BaseModel):
    flights: list[Flight]


def flight_data_to_pandas(flight_data: FlightData) -> pd.DataFrame:
    flight_data_dict = {"OPERA": [], "TIPOVUELO": [], "MES": []}
    for elem in flight_data.flights:
        flight_data_dict["OPERA"].append(elem.OPERA)
        flight_data_dict["TIPOVUELO"].append(elem.TIPOVUELO)
        flight_data_dict["MES"].append(elem.MES)

    return pd.DataFrame(flight_data_dict)


@app.get("/", status_code=200)
async def root() -> dict:
    return {
        "message": (
            "welcome to the api for predicting flight delay. Use the /health "
            "endpoint to get server status, and the /predict endpoint to get your "
            "prediction from input data."
        )
    }


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> dict:
    try:
        # get data and convert to pandas dataframe
        flight_data_df = flight_data_to_pandas(flight_data)
        preprocessed_data = model.preprocess(flight_data_df)

        # sorts column to feed the model
        column_order = model._model.feature_names_in_
        preprocessed_data = preprocessed_data[column_order]

        pred = model.predict(preprocessed_data)

        return {"predict": pred}
    except Exception as e:
        # there may be exceptions we don't want to send to the clients, so log them in
        # an internal file for debugging. Just as a cheap solution.
        with open("error_logs.txt", "a") as f:
            f.write(f"{datetime.now(timezone.utc)}: encounter error {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )
