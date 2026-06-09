from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model_service import load_model, predict_blue_win_probability


class TeamComposition(BaseModel):
    top: str = Field(..., examples=["Garen"])
    jungle: str = Field(..., examples=["Skarner"])
    mid: str = Field(..., examples=["Cassiopeia"])
    adc: str = Field(..., examples=["Caitlyn"])
    support: str = Field(..., examples=["Zilean"])


class PredictionRequest(BaseModel):
    blue_team: TeamComposition
    red_team: TeamComposition
    blue_team_win_rate: Optional[float] = Field(default=0.5, ge=0, le=1)
    red_team_win_rate: Optional[float] = Field(default=0.5, ge=0, le=1)
    blue_players_with_win_rate: Optional[int] = Field(default=0, ge=0, le=5)
    red_players_with_win_rate: Optional[int] = Field(default=0, ge=0, le=5)


app = FastAPI(
    title="LoL Blue Team Win Probability API",
    description="API para estimar a probabilidade de vitoria do time azul com base na composicao dos times.",
)

model_artifact = load_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    return predict_blue_win_probability(request.model_dump(), model_artifact)
