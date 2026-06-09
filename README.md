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

## Observacoes

Os notebooks foram deixados fora do commit inicial por enquanto. Chaves da Riot API nao devem ser versionadas; use variaveis de ambiente, como `RIOT_API_KEY`.
