from pathlib import Path
from typing import Any

import joblib
import pandas as pd
"""
Funções para carregar o modelo, preparar os dados de entrada e fazer previsões. 
O modelo é treinado para estimar a probabilidade de vitória do time azul com base na composição dos times e outras características relevantes. 
O código inclui tratamento para recursos numéricos, criação de recursos binários para a presença de campeões específicos e verificação de campeões desconhecidos em relação ao pool de campeões usado no treinamento do modelo.
"""

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "blue_win_model.joblib"

ROLE_CHAMPION_FEATURES = [
    "blue_top_champion",
    "blue_jungle_champion",
    "blue_mid_champion",
    "blue_adc_champion",
    "blue_support_champion",
    "red_top_champion",
    "red_jungle_champion",
    "red_mid_champion",
    "red_adc_champion",
    "red_support_champion",
]

BLUE_ROLE_FEATURES = [column for column in ROLE_CHAMPION_FEATURES if column.startswith("blue_")]
RED_ROLE_FEATURES = [column for column in ROLE_CHAMPION_FEATURES if column.startswith("red_")]

DEFAULT_NUMERIC_FEATURES = [
    "blue_team_win_rate",
    "red_team_win_rate",
    "blue_players_with_win_rate",
    "red_players_with_win_rate",
    "team_win_rate_diff",
]


def _safe_champion_name(champion: str) -> str:
    return champion.replace(" ", "_").replace("'", "").replace(".", "")


def _prepare_numeric_features(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    df = df.copy()

    if "blue_team_win_rate" in numeric_features:
        df["blue_team_win_rate"] = pd.to_numeric(df.get("blue_team_win_rate", 0.5), errors="coerce").fillna(0.5)
    if "red_team_win_rate" in numeric_features:
        df["red_team_win_rate"] = pd.to_numeric(df.get("red_team_win_rate", 0.5), errors="coerce").fillna(0.5)
    if "blue_players_with_win_rate" in numeric_features:
        df["blue_players_with_win_rate"] = (
            pd.to_numeric(df.get("blue_players_with_win_rate", 0), errors="coerce").fillna(0)
        )
    if "red_players_with_win_rate" in numeric_features:
        df["red_players_with_win_rate"] = (
            pd.to_numeric(df.get("red_players_with_win_rate", 0), errors="coerce").fillna(0)
        )
    if "team_win_rate_diff" in numeric_features:
        blue_win_rate = pd.to_numeric(df.get("blue_team_win_rate", 0.5), errors="coerce").fillna(0.5)
        red_win_rate = pd.to_numeric(df.get("red_team_win_rate", 0.5), errors="coerce").fillna(0.5)
        df["team_win_rate_diff"] = blue_win_rate - red_win_rate

    return df


def _build_side_champion_features(df: pd.DataFrame, champion_pool: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    side_feature_data = {}
    side_champion_features = []

    for champion in champion_pool:
        safe_champion = _safe_champion_name(champion)
        blue_column = f"blue_has_{safe_champion}"
        red_column = f"red_has_{safe_champion}"

        side_feature_data[blue_column] = df[BLUE_ROLE_FEATURES].eq(champion).any(axis=1).astype(int)
        side_feature_data[red_column] = df[RED_ROLE_FEATURES].eq(champion).any(axis=1).astype(int)
        side_champion_features.extend([blue_column, red_column])

    df = pd.concat([df, pd.DataFrame(side_feature_data, index=df.index)], axis=1)
    return df, side_champion_features


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute o notebook modelo_vitoria_time_azul.ipynb "
            f"e salve o artefato em {MODEL_PATH}."
        )

    artifact = joblib.load(MODEL_PATH)
    if hasattr(artifact, "predict_proba"):
        raise ValueError(
            "O arquivo do modelo contém apenas o pipeline. Reexecute a última célula do notebook "
            "para salvar o artefato completo com pipeline, champion_pool e feature_columns."
        )

    return artifact


def build_prediction_frame(composition: dict[str, Any], artifact: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    champion_pool = artifact["champion_pool"]
    numeric_features = artifact.get("numeric_features", DEFAULT_NUMERIC_FEATURES)
    row = {
        "blue_top_champion": composition["blue_team"]["top"],
        "blue_jungle_champion": composition["blue_team"]["jungle"],
        "blue_mid_champion": composition["blue_team"]["mid"],
        "blue_adc_champion": composition["blue_team"]["adc"],
        "blue_support_champion": composition["blue_team"]["support"],
        "red_top_champion": composition["red_team"]["top"],
        "red_jungle_champion": composition["red_team"]["jungle"],
        "red_mid_champion": composition["red_team"]["mid"],
        "red_adc_champion": composition["red_team"]["adc"],
        "red_support_champion": composition["red_team"]["support"],
        "blue_team_win_rate": composition.get("blue_team_win_rate", 0.5),
        "red_team_win_rate": composition.get("red_team_win_rate", 0.5),
        "blue_players_with_win_rate": composition.get("blue_players_with_win_rate", 0),
        "red_players_with_win_rate": composition.get("red_players_with_win_rate", 0),
    }

    df = pd.DataFrame([row])
    df = _prepare_numeric_features(df, numeric_features)
    df, _ = _build_side_champion_features(df, champion_pool)

    unknown_champions = sorted(
        {
            champion
            for champion in df[ROLE_CHAMPION_FEATURES].iloc[0].tolist()
            if champion not in champion_pool
        }
    )

    return df[artifact["feature_columns"]], unknown_champions


def predict_blue_win_probability(composition: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    X, unknown_champions = build_prediction_frame(composition, artifact)
    probability = float(artifact["pipeline"].predict_proba(X)[0, 1])

    return {
        "blue_win_probability": probability,
        "red_win_probability": 1 - probability,
        "unknown_champions": unknown_champions,
        "model_features": len(artifact["feature_columns"]),
    }
