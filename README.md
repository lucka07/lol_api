# lol_api

Projeto para coletar partidas ranqueadas de League of Legends pela Riot API e desenvolver modelos para estimar a probabilidade de vitoria do time azul.

## Estrutura

- `dataset/`: dados gerados a partir das partidas coletadas.
- `src/`: codigo fonte do projeto.
- `models/`: modelos treinados e artefatos de machine learning.

## Dados

O dataset principal esperado e:

```text
dataset/matches.csv
```

Ele contem informacoes das partidas, composicoes dos times, resultado do time azul e features auxiliares como win rate medio conhecido dos jogadores.

## API de predicao

Instale as dependencias e suba a API:

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Antes de iniciar a API, execute o notebook `modelo_vitoria_time_azul.ipynb` e salve o modelo em:

```text
models/blue_win_model.joblib
```

Ao iniciar, a API carrega esse artefato salvo pelo notebook.

Endpoint:

```text
POST /predict
```

Exemplo de payload:

```json
{
  "blue_team": {
    "top": "Garen",
    "jungle": "Skarner",
    "mid": "Cassiopeia",
    "adc": "Caitlyn",
    "support": "Zilean"
  },
  "red_team": {
    "top": "Malphite",
    "jungle": "RekSai",
    "mid": "Kassadin",
    "adc": "KogMaw",
    "support": "Milio"
  }
}
```

Resposta:

```json
{
  "blue_win_probability": 0.6652,
  "red_win_probability": 0.3348,
  "unknown_champions": [],
  "model_features": 349
}
```

## Observacoes

Os notebooks foram deixados fora do commit inicial por enquanto. Chaves da Riot API nao devem ser versionadas; use variaveis de ambiente, como `RIOT_API_KEY`.
