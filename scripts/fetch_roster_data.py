import requests
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("CFBD_API_KEY")


# debug
print(API_KEY)
# change these to target different team, year and filename
TEAM_TARGET = "Georgia"
YEAR_TARGET = 2025
FILE_TARGET = "data/uga/uga_roster_2025.csv"

headers = {"Authorization": f"Bearer {API_KEY}"}
params = {"team": TEAM_TARGET, "year": YEAR_TARGET}

r = requests.get(
    "https://api.collegefootballdata.com/roster",
    headers=headers,
    params=params
)

print(r.status_code)
data = r.json()
print(data[:2])  # peek at first 2 records

df = pd.DataFrame(data)
df.to_csv(FILE_TARGET, index=False)
print(f"Saved {len(df)} players")
print(df.columns.tolist())