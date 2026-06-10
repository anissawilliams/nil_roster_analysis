import requests
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

TEAM_TARGET = "Georgia"
YEAR_TARGET = 2025
FILE_TARGET = "data/uga/uga_transfers_2025.csv"


API_KEY = os.environ.get("CFBD_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

r = requests.get(
    "https://api.collegefootballdata.com/player/portal",
    headers=headers,
    params={"year": YEAR_TARGET}
)

df = pd.DataFrame(r.json())
print(f"Total portal entries: {len(df)}")

# filter transfers to just the team we care about
transfers_in  = df[df["destination"] == TEAM_TARGET]
transfers_out = df[df["origin"] == TEAM_TARGET]

print(f"Incoming to : {TEAM_TARGET}  {len(transfers_in)}")
print(f"Outgoing from : {TEAM_TARGET}  {len(transfers_out)}")

print("\nIncoming sample:")
print(transfers_in.head(3).to_string())

print("\nOutgoing sample:")
print(transfers_out.head(3).to_string())

# save both
transfers_in["direction"]  = "Incoming"
transfers_out["direction"] = "Outgoing"
transfers = pd.concat([transfers_in, transfers_out])
transfers.to_csv(FILE_TARGET, index=False)
print(f"\nSaved {len(transfers)} {TEAM_TARGET} transfer entries")