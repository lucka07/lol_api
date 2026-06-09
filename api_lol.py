import requests
import time
import os

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise ValueError("Defina a variável de ambiente RIOT_API_KEY antes de executar.")

PLATFORM = "br1"

headers = {
    "X-Riot-Token": API_KEY
}

url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"

response = requests.get(url, headers=headers)
challenger_data = response.json()

entries = challenger_data["entries"]

print(len(entries))
